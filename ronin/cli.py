"""ronin CLI: init, doctor, gpus, prices, profiles, volume, up, down, status, cost, logs,
watchdog, env, opencode."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import __version__, catalog, config, fit, mascot, ui
from .api import ProviderError
from .config import Config, ConfigError, load_config, load_env, load_state, save_state
from .pod import (CONTAINER_DISK_PRICE_GB_MONTH, CONTAINER_START, DEAD_STATES, FATAL_MARKERS,
                  NO_GPU_MARKERS, OPENCODE_API_KEY_ENV, PROVIDERS, VOLUME_POD_DISK_GB, LogSnapshot, LogStream,
                  clear_pod, client, claude_env, endpoint, ensure_volume, guard_text, healthy,
                  make_body, max_model_len, monthly, opencode_config, orphans, pod_problem,
                  pod_provider, real_session_cost, remember_pod, root_cause, session_cost,
                  terminate, terminate_own, vast_offer)
from .watchdog import WATCHDOG_POLL, cmd_watchdog

# Waiting for the container to come up. vast.ai hosts pull the server image themselves
# (RunPod's are cached), which can take a while on a slow host.
POD_READY_TIMEOUT = {"runpod": 10 * 60, "vast": 30 * 60}


TUNNEL_WAIT = 90                 # seconds to wait for a vast.ai pod's HTTPS tunnel once ready


# The guard logs the URL of its HTTPS tunnel (vast.ai pods; see guard.py).
TUNNEL_LINE = re.compile(r"\[ronin-guard\] tunnel (https://\S+)")


MODEL_READY_TIMEOUT = 45 * 60    # weight download (always in ephemeral, first time with a volume)


WARN_AFTER_HOURS = 4             # status warns about a long-running (billing) session


SNAPSHOT_EVERY = 20              # seconds between fresh log snapshots while `up` waits


FATAL_GRACE = 15                 # seconds to collect the traceback after the marker


# Startup phase from the latest recognised log line (in order of appearance).
PHASES = (
    ("Pulling from", "Pulling the server image"),
    (CONTAINER_START, "Starting the server"),
    # vLLM
    ("Loading model from scratch", "Downloading the model weights"),
    ("Loading safetensors checkpoint shards", "Loading the weights onto the GPU"),
    ("Dynamo bytecode transform", "Compiling the model"),
    ("torch.compile took", "Warming up the model"),
    ("Capturing CUDA graphs", "Capturing CUDA graphs"),
    ("Starting vLLM server", "Opening the server"),
    # llama.cpp
    ("[ronin-guard] installing python3", "Installing the guard's Python"),
    ("common_download_file", "Downloading the GGUF"),
    ("llama_model_loader: loaded meta data", "Reading the GGUF"),
    ("load_tensors:", "Loading the weights onto the GPU"),
    ("server is listening", "Opening the server"),
)


# --- commands ---

def cmd_init(args, cfg: Config | None) -> None:
    """Set up the project in the current directory (or the one --config points at)."""
    # Not the discovered project: a ronin.toml up the tree or in ronin's own checkout
    # is not where a new project should be created.
    target = args.config or os.environ.get("RONIN_CONFIG") or Path.cwd() / config.CONFIG_NAME
    project = config.project = config.Project(Path(target))
    load_env()
    if project.config.exists():
        ui.ok(f"Config found: {project.config}")
    else:
        project.config.write_text(config.STARTER_CONFIG, encoding="utf-8")
        ui.ok(f"Created {project.config}")
        ui.hint("Edit the `main` profile's model and GPUs, or add your own profiles.")
    if project.env.exists():
        ui.ok(f"Secrets file found: {project.env}")
    else:
        project.env.write_text(config.STARTER_ENV, encoding="utf-8")
        ui.ok(f"Created {project.env}")
    ignore = project.root / ".gitignore"
    if ignore.exists():
        text = ignore.read_text(encoding="utf-8")
        missing = [entry for entry in (".env", ".ronin/") if entry not in text.splitlines()]
        if missing:
            with ignore.open("a", encoding="utf-8") as f:
                f.write(("" if not text or text.endswith("\n") else "\n") + "\n".join(missing) + "\n")
            ui.ok(f"Added {', '.join(missing)} to .gitignore (secrets and local state)")
    cfg = load_config()
    key = PROVIDERS[cfg.provider][1]
    if not os.environ.get(key):
        print()
        ui.warn(f"{key} is empty")
        ui.hint(f"Paste your key into {project.env} ({KEY_PAGES[cfg.provider]}), "
                "then run `ronin doctor`.")
        return
    if cfg.storage == "volume":
        vol_id = ensure_volume(client(), cfg, load_state())
        ui.ok(f"Network volume ready: {vol_id} ({cfg.datacenter})")
    print()
    ui.hint("Next: `ronin doctor` to check everything, then `ronin up`.")


KEY_PAGES = {"runpod": "https://console.runpod.io/user/settings",
             "vast": "https://cloud.vast.ai/manage-keys/"}


def cmd_doctor(args, cfg: Config | None) -> None:
    """Check the setup end to end and report every problem, not just the first."""
    problems = 0

    def bad(msg: str, hint: str | None = None) -> None:
        nonlocal problems
        problems += 1
        ui.error(f"{msg}\n{hint}" if hint else msg)

    project = config.project
    try:
        cfg = load_config()
    except ConfigError as e:
        bad(str(e))
        cfg = None
    else:
        ui.ok(f"Config: {project.config} {ui.dim(f'· {len(cfg.profiles)} profile(s) · default {cfg.default_profile}')}")
        for p in cfg.profiles.values():
            if p.chat_template and not (project.root / p.chat_template).exists():
                bad(f"Profile {p.name}: chat template not found", str(project.root / p.chat_template))
        if cfg.storage == "volume" and not cfg.datacenter:
            bad('storage = "volume" needs a datacenter', "`ronin gpus` shows where there is stock.")
        guard = [f"{cfg.max_hours:g} h max" if cfg.max_hours > 0 else "",
                 f"{cfg.idle_minutes:g} min idle" if cfg.idle_minutes > 0 else ""]
        if any(guard):
            ui.ok(f"Guard: the pod deletes itself after {' or '.join(g for g in guard if g)}")
        else:
            ui.warn("Guard disabled (max_hours and idle_minutes are 0): a forgotten pod bills forever")

    provider = cfg.provider if cfg else "runpod"
    key = PROVIDERS[provider][1]
    label = PROVIDERS[provider][0].label
    rp = None
    if not os.environ.get(key):
        bad(f"{key} is not set", f"Add it to {project.env} (`ronin init` creates the file).")
    else:
        rp = client(provider)
        try:
            gpus = rp.list_gpus(cfg.cloud if cfg else "SECURE")
        except ProviderError as e:
            hint = "Check the key in .env." if e.status in (401, 403) else None
            bad(f"{label} API: {e.detail}", hint)
            rp = None
        else:
            ui.ok(f"{label} API key works")
            try:
                balance = rp.account()["balance"]
            except ProviderError:
                balance = None
            if balance is not None and balance <= 0:
                bad(f"{label} balance is ${balance:.2f}: it will not start a pod",
                    "Add credit in the provider's billing page.")
            if cfg and provider == "vast":
                for p in cfg.profiles.values():
                    for gpu in p.gpus:
                        try:
                            catalog.provider_gpu(gpu, "vast")
                        except ConfigError as e:
                            bad(f"Profile {p.name}: {e}")
            if cfg:
                profile = cfg.profile(None)
                # the datacenter is RunPod's; vast.ai lists countries there
                dc = cfg.datacenter if provider == "runpod" else ""
                stock = {catalog.key(g["id"]): g for g in gpus}
                available = [g for g in profile.gpus
                             if stock.get(catalog.key(g), {}).get("availability") not in (None, "NONE")
                             and (not dc or dc in [d["id"] for d in
                                                   stock.get(catalog.key(g), {}).get("dataCenters", [])])]
                where = dc or ("any datacenter" if provider == "runpod" else label)
                if available:
                    ui.ok(f"Stock for profile {profile.name}: {', '.join(available)} {ui.dim('in ' + where)}")
                else:
                    ui.warn(f"No stock right now for profile {profile.name}'s GPUs in {where}")
                    ui.hint("`ronin gpus` to see alternatives.")
                for p in cfg.profiles.values():
                    doctor_memory(p, gpus, bad)

    if cfg and cfg.client == "claude":
        agent_bin, agent_label, agent_url = "claude", "Claude Code", "https://claude.com/claude-code"
    else:
        agent_bin = cfg.opencode_bin if cfg else "opencode"
        agent_label, agent_url = "opencode", "https://opencode.ai"
    if shutil.which(agent_bin):
        ui.ok(f"{agent_label}: {shutil.which(agent_bin)}")
    else:
        bad(f"`{agent_bin}` is not on PATH", f"Install {agent_label}: {agent_url}")

    state = load_state()
    if state.get("pod_id"):
        ui.warn(f"A pod is registered: {state['pod_id']} · `ronin status` to see what it costs")
    doctor_orphans(state, bad)

    print()
    if problems:
        raise ConfigError(f"{problems} problem{'s' if problems > 1 else ''} found")
    ui.ok("All good. `ronin up` to start.")


CONSOLES = {"runpod": "https://www.runpod.io/console/pods",
            "vast": "https://cloud.vast.ai/instances/"}


def doctor_orphans(state: dict, bad) -> None:
    """Pods ronin named that this project does not know: they bill unseen."""
    for rp, p in orphans(state):
        cost = p.get("cost")
        bad(f"{rp.label} pod {p['id']} ({p.get('name')}, {p.get('status')}) is not registered "
            f"here and may be billing ≈ ${cost if isinstance(cost, (int, float)) else 0:.2f}/h",
            f"Delete it at {CONSOLES[rp.name]} if it is not another project's.")


def doctor_memory(profile, catalog: list[dict], bad) -> None:
    if profile.backend != "vllm":
        ui.step(f"Profile {profile.name}: {profile.backend} / GGUF — memory check skipped "
                f"{ui.dim('(no safetensors metadata; size the quant yourself)')}")
        return
    try:
        params = fit.checkpoint_params(profile.model)
    except LookupError as e:
        ui.warn(f"Profile {profile.name}: memory check skipped ({e})")
        return
    weights = fit.weights_gib(params, profile)
    results = fit.fits(profile, weights, catalog)
    what = f"~{weights:.0f} GiB of weights ({fit.precision(params, profile)})"
    if not any(f.fits for f in results):
        bad(f"Profile {profile.name}: {what} fit on none of its GPUs",
            fit.too_small_hint(profile, params))
        return
    ui.ok(f"Profile {profile.name}: {what} fit")
    for f in results:
        if not f.fits:
            ui.warn(f"Profile {profile.name}: {f.gpu} is too small ({f.vram_gb:g} GB); `ronin up` skips it")
        elif f.tight:
            ui.warn(f"Profile {profile.name}: on {f.gpu} only ~{f.spare_gib:.0f} GiB is left for the KV cache")


def memory_check(rp, cfg: Config, profile) -> list[fit.Fit]:
    """The profile's GPUs that can hold the weights, in order; warns about the rest.
    Stops `ronin up` before any pod is created (and billed) when none can."""
    unchecked = [fit.Fit(g, None, None) for g in profile.gpus]
    if profile.backend != "vllm":
        ui.step(f"Memory check skipped: {profile.backend} / GGUF has no safetensors size")
        return unchecked
    try:
        params = fit.checkpoint_params(profile.model)
        catalog = rp.list_gpus(cfg.cloud)
    except (LookupError, ProviderError) as e:
        ui.warn(f"Memory check skipped: {getattr(e, 'detail', e)}")
        return unchecked
    weights = fit.weights_gib(params, profile)
    results = fit.fits(profile, weights, catalog)
    what = f"~{weights:.0f} GiB of weights ({fit.precision(params, profile)})"
    usable = [f for f in results if f.fits]
    if not usable:
        raise ConfigError(f"The model does not fit on any GPU of profile {profile.name}\n"
                          f"{what}, plus ~{fit.OVERHEAD_GIB} GiB for vLLM and the KV cache.\n"
                          f"{fit.too_small_hint(profile, params)}\n"
                          "No pod was created. `ronin up --skip-memory-check` to try anyway.")
    ui.step(f"Memory: {what}")
    for f in results:
        if not f.fits:
            ui.warn(f"Skipping {f.gpu}: {f.vram_gb:g} GB cannot hold the weights")
    return usable


def cmd_volume(args, cfg: Config) -> None:
    {"ls": volume_ls, "create": volume_create, "rm": volume_rm}[args.action](args, cfg)


def volume_ls(args, cfg: Config) -> None:
    volumes = client("runpod").list_volumes()
    if not volumes:
        ui.step("No network volumes.")
        if cfg.storage == "volume":
            ui.hint("`ronin volume create` to create the one in ronin.toml")
        return
    current = load_state().get("volume_id")
    w_id = max(len(v["id"]) for v in volumes)
    w_name = max(len(v["name"]) for v in volumes)
    print("  " + ui.dim(f"{'id':<{w_id}}  {'name':<{w_name}}  {'datacenter':<10}  {'size':>7}  cost"))
    for v in volumes:
        size = v.get("size", 0)
        row = (f"{v['id']:<{w_id}}  {v['name']:<{w_name}}  {v['dataCenter']:<10}  "
               f"{str(size) + ' GB':>7}  ≈ {monthly(size)}")
        print(f"  {ui.bold(row) if v['id'] == current else row}")
    total = sum(v.get("size", 0) for v in volumes)
    print()
    ui.hint(f"Total ≈ {monthly(total)} · bold: this project's volume · `ronin volume rm <id>` to delete one")


def volume_create(args, cfg: Config) -> None:
    if cfg.storage != "volume":
        ui.warn('storage = "ephemeral" in ronin.toml: the volume would not be used')
        ui.hint('Set storage = "volume" first.')
        return
    vol_id = ensure_volume(client("runpod"), cfg, load_state())
    ui.ok(f"Network volume ready: {vol_id} ({cfg.datacenter})")


def volume_rm(args, cfg: Config) -> None:
    rp = client("runpod")
    vol = next((v for v in rp.list_volumes() if v["id"] == args.id), None)
    if vol is None:
        raise ConfigError(f"No network volume with id {args.id}\n`ronin volume ls` lists them.")
    what = f"{vol['name']} ({vol['dataCenter']}, {vol.get('size', '?')} GB)"
    if not args.yes:
        if not sys.stdin.isatty():
            raise ConfigError("Refusing to delete a volume without confirmation\nPass --yes.")
        ui.warn(f"This deletes volume {args.id} {what} and every model stored on it.")
        if input("    Type the volume id to confirm: ").strip() != args.id:
            ui.step("Cancelled.")
            return
    rp.delete_volume(args.id)
    state = load_state()
    if state.get("volume_id") == args.id:
        state.pop("volume_id")
        save_state(state)
    ui.ok(f"Volume {args.id} deleted: {what}")


def cmd_profiles(args, cfg: Config) -> None:
    rows = []
    for name, p in cfg.profiles.items():
        gpu = p.gpus[0] + (f" x{p.gpu_count}" if p.gpu_count > 1 else "")
        backend = "llama.cpp" if p.backend == "llamacpp" else "vLLM"
        rows.append((name + (" *" if name == cfg.default_profile else ""),
                     gpu, backend, f"{max_model_len(p) // 1024}K", p.model))
    w_name = max(len(r[0]) for r in rows)
    w_gpu = max(len(r[1]) for r in rows)
    w_be = max(len(r[2]) for r in rows)
    w_name = max(w_name, len("profile"))
    w_gpu = max(w_gpu, len("GPU"))
    w_be = max(w_be, len("serve"))
    print("  " + ui.dim(f"{'profile':<{w_name}}  {'GPU':<{w_gpu}}  {'serve':<{w_be}}  {'ctx':>5}  model"))
    for name, gpu, backend, ctx, model in rows:
        # pad before colouring: ANSI codes would count towards the width
        print(f"  {ui.bold(name.ljust(w_name))}  {gpu:<{w_gpu}}  {backend:<{w_be}}  "
              f"{ctx:>5}  {ui.dim(model)}")
    print()
    ui.hint("* default profile · `ronin up -p <profile>` to pick another")


def cmd_gpus(args, cfg: Config) -> None:
    cloud_name = "SECURE" if args.secure else cfg.cloud
    cloud = cloud_name.lower()
    rp = client(args.provider or cfg.provider)
    gpus = [g for g in rp.list_gpus(cloud_name) if g.get("memory", 0) >= args.min_vram]
    gpus.sort(key=lambda g: (g.get("price", {}).get(cloud) or 0))
    if not gpus:
        ui.warn(f"No GPUs with {args.min_vram} GB or more in {cloud_name}.")
        return

    stock_color = {"HIGH": ui.green, "MEDIUM": ui.yellow, "LOW": ui.yellow}
    rows = []
    for g in gpus:
        price = g.get("price", {}).get(cloud)
        dcs = [dc["id"] for dc in g.get("dataCenters", [])]
        rows.append((g["id"], f"{g['memory']} GB", f"${price:.2f}" if price is not None else "-",
                     g.get("availability", "-"), dcs))
    w_id = max(len(r[0]) for r in rows)

    where = "datacenters" if rp.name == "runpod" else "countries"
    print(f"  {ui.bold(f'GPUs with {args.min_vram} GB or more')} "
          f"{ui.dim(f'· {rp.label} · {cloud_name} cloud · price per hour')}\n")
    print("  " + ui.dim(f"{'GPU':<{w_id}}  {'VRAM':>6}  {'$/h':>6}  {'stock':<6}  {where}"))
    for gpu_id, vram, price, stock, dcs in rows:
        color = stock_color.get(stock, ui.dim)
        dc_text = ", ".join(ui.bold(d) if d == cfg.datacenter else d for d in dcs) or ui.dim("-")
        print(f"  {gpu_id:<{w_id}}  {vram:>6}  {price:>6}  {color(f'{stock:<6}')}  {dc_text}")
    if rp.name == "vast":
        print()
        ui.hint("Stock = verified offers (HIGH 10+, MEDIUM 3+) · GPU names as profiles write them")
    elif cfg.datacenter:
        print()
        ui.hint(f"Bold: your datacenter ({cfg.datacenter}).")


@dataclass
class Quote:
    """What one GPU of a profile costs on a provider right now."""
    gpu: str                        # as the profile names it
    stock: str | None = None        # None: cannot be had now (no offers / not offered)
    gpu_h: float = 0                # $/h of the GPUs (gpu_count of them)
    disk_h: float = 0               # $/h of the pod's disk
    per_gb: float = 0               # $/GB of the weight download
    note: str = ""
    hourly_max: float = 0           # vast.ai: the dearest offer `up` may fall back to
    per_gb_max: float = 0

    @property
    def hourly(self) -> float:
        return self.gpu_h + self.disk_h

    def session(self, hours: float, download_gb: float | None) -> float:
        """$/h × hours plus one download of the weights (unknown size counts as 0)."""
        return self.hourly * hours + (download_gb or 0) * self.per_gb


QUOTE_WORKERS = 4                # parallel offer searches for `ronin prices`


HOURS_PER_MONTH = 30 * 24        # how providers turn $/GB/month into $/h


def runpod_quotes(rp, cfg: Config, profiles, cloud: str) -> dict[str, list[Quote]]:
    """RunPod: the catalog's per-GPU price × gpu_count, plus the container disk.
    Ingress (the weight download) is free on RunPod."""
    gpus = {catalog.key(g["id"]): g for g in rp.list_gpus(cloud)}
    quotes = {}
    for p in profiles:
        disk = VOLUME_POD_DISK_GB if cfg.storage == "volume" else p.disk_gb
        rows = []
        for gpu in p.gpus:
            g = gpus.get(catalog.key(gpu), {})
            q = Quote(gpu, disk_h=disk * CONTAINER_DISK_PRICE_GB_MONTH / HOURS_PER_MONTH)
            price = (g.get("price") or {}).get(cloud.lower())
            stock = g.get("availability")
            dcs = [d["id"] for d in g.get("dataCenters", [])]
            if price is not None:
                q.gpu_h = price * p.gpu_count       # the list price, with stock or not
            if price is None or stock in (None, "NONE"):
                q.note = "no stock"
            elif cfg.datacenter and cfg.datacenter not in dcs:
                q.note = f"no stock in {cfg.datacenter}"
            else:
                q.stock = stock
            rows.append(q)
        quotes[p.name] = rows
    return quotes


def vast_quotes(rp, cfg: Config, profiles, cloud: str) -> dict[str, list[Quote]]:
    """vast.ai: the offer `up` would rent now (ronin's filters, the profile's disk), with
    its storage and download prices. One search per distinct GPU/count/disk/CUDA."""
    asks = {}                       # (profile, gpu) -> filters, or the ConfigError text
    for p in profiles:
        for gpu in p.gpus:
            try:
                asks[(p.name, gpu)] = vast_offer(cfg, p, gpu, cloud)
            except ConfigError as e:
                asks[(p.name, gpu)] = str(e).splitlines()[0]
    disks = {p.name: p.disk_gb for p in profiles}
    searches = {json.dumps([f, disks[name]], sort_keys=True): (f, disks[name])
                for (name, _), f in asks.items() if isinstance(f, dict)}
    with ThreadPoolExecutor(QUOTE_WORKERS) as pool:
        found = dict(zip(searches, pool.map(lambda fd: rp.quote(*fd), searches.values())))
    quotes = {}
    for p in profiles:
        rows = []
        for gpu in p.gpus:
            ask = asks[(p.name, gpu)]
            if not isinstance(ask, dict):
                rows.append(Quote(gpu, note="not on vast.ai"))
                continue
            offer = found[json.dumps([ask, p.disk_gb], sort_keys=True)]
            if not offer:
                rows.append(Quote(gpu, note="no offers"))
                continue
            n = offer["offers"]
            rows.append(Quote(gpu, stock=f"{n}{'+' if n >= 50 else ''} offer{'s' if n > 1 else ''}",
                              gpu_h=offer["gpu"], disk_h=offer["disk"], per_gb=offer["download"],
                              hourly_max=offer["hourly_max"], per_gb_max=offer["download_max"]))
        quotes[p.name] = rows
    return quotes


def cmd_prices(args, cfg: Config) -> None:
    """Current price of each profile's GPUs on a provider: GPU, disk, the weight download
    and the total for a session of --hours."""
    provider = args.provider or cfg.provider
    cloud = "SECURE" if args.secure else cfg.cloud
    names = args.profiles or list(cfg.profiles)
    profiles = [cfg.profile(n) for n in names]
    rp = client(provider)
    quotes = (vast_quotes if provider == "vast" else runpod_quotes)(rp, cfg, profiles, cloud)
    cached = provider == "runpod" and cfg.storage == "volume"   # weights already on the volume
    sizes = {}
    for p in profiles:
        try:
            sizes[p.name] = 0.0 if cached else fit.download_gb(p)
        except LookupError:
            sizes[p.name] = None

    hours = args.hours
    head = (f"{'profile':<16}  {'GPU':<28}  {'GPU/h':>7}  {'disk/h':>6}  {'$/h':>7}  "
            f"{'download':>16}  {f'{hours:g} h total':>10}  stock")
    print(f"  {ui.bold('Prices for your profiles')} "
          f"{ui.dim(f'· {rp.label} · {cloud} cloud · now')}\n")
    print("  " + ui.dim(head))
    for p in profiles:
        picked = next((q for q in quotes[p.name] if q.stock), None)
        gb = sizes[p.name]
        for i, q in enumerate(quotes[p.name]):
            # pad before colouring: ANSI codes would count towards the width
            name = (p.name if i == 0 else "").ljust(16)
            mark = ui.SYM["pick"] + " " if q is picked else "  "
            gpu = (catalog.key(q.gpu) + (f" x{p.gpu_count}" if p.gpu_count > 1 else ""))[:26].ljust(26)
            if not q.gpu_h:                  # not even a price: nothing to show
                print(f"  {ui.bold(name)}  {mark}{ui.dim(gpu)}  {ui.dim(q.note)}")
                continue
            if gb is None:
                download = "? GB".rjust(16)
            elif cached:
                download = "on volume".rjust(16)
            else:
                cost = "free" if gb * q.per_gb < 0.005 else f"${gb * q.per_gb:.2f}"
                download = f"{cost} · {gb:.0f} GB".rjust(16)
            total = f"${q.session(hours, gb):.2f}" + ("+" if gb is None and q.per_gb else "")
            cells = (f"{gpu}  {'$' + format(q.gpu_h, '.2f'):>7}  {'$' + format(q.disk_h, '.3f'):>6}  "
                     f"{'$' + format(q.hourly, '.2f'):>7}  {download}  ")
            if q.stock:
                print(f"  {ui.bold(name)}  {mark}{cells}"
                      f"{ui.bold(total.rjust(10)) if q is picked else total.rjust(10)}  {ui.dim(q.stock)}")
            else:                             # priced, but cannot be had right now
                print(f"  {ui.bold(name)}  {mark}{ui.dim(cells + total.rjust(10))}  {ui.yellow(q.note)}")
    print()
    ui.hint(f"{ui.SYM['pick']} the GPU `ronin up` takes: the first in the profile's order with stock")
    ui.hint(f"Total = $/h × {hours:g} h (--hours) + one download of the weights; the "
            "download minutes also bill at $/h")
    if provider == "vast":
        ui.hint("vast.ai: the offer `up` would rent now; the host bills the download per GB")
    elif cached:
        ui.hint(f"Weights on the network volume: no download, but the volume bills "
                f"≈ {monthly(cfg.volume_size_gb)} whether a pod runs or not")
    else:
        ui.hint("RunPod: container disk ≈ $0.10/GB/month while running; downloads are free")


def cmd_up(args, cfg: Config) -> None:
    profile = cfg.profile(args.profile)
    provider = getattr(args, "provider", None) or cfg.provider
    config.check_provider(provider, cfg.storage)
    if provider == "vast":
        # ronin.toml was checked for its own provider: with --provider vast, check the
        # GPU names now, before anything is created
        for gpu in profile.gpus:
            catalog.provider_gpu(gpu, "vast")
    rp = client(provider)
    state = load_state()

    waiting = waiting_for_stock(state)
    if waiting:
        # two waiting `up`s could each create a pod, and both would bill
        ui.warn("Another " + waiting_text(waiting))
        ui.hint("Wait for it, or Ctrl+C it in its terminal first.")
        return

    if state.get("pod_id"):
        # the registered pod may be on another provider than this one
        old_rp = client(pod_provider(state))
        try:
            pod = old_rp.get_pod(state["pod_id"])
        except ProviderError as e:
            if e.status != 404:
                raise
            clear_pod(state)
        else:
            if pod["status"] not in DEAD_STATES:
                ui.warn(f"A pod is already active: {pod['id']} ({pod['status']}) "
                        f"· profile {state.get('profile')} · {old_rp.label}")
                ui.hint("`ronin status` to see it · `ronin down` to delete it")
                return
            # A stopped pod still exists (and keeps its disk): delete it, do not orphan it.
            ui.step(f"Deleting the previous pod {pod['id']} ({pod['status']})")
            terminate(old_rp, state)

    ui.step(f"Profile {ui.bold(profile.name)} {ui.dim('· ' + profile.model + ' · ' + rp.label)}")
    if args.skip_memory_check:
        candidates = [fit.Fit(g, None, None) for g in profile.gpus]
    else:
        candidates = memory_check(rp, cfg, profile)
    volume_id = ensure_volume(rp, cfg, state) if cfg.storage == "volume" else None
    api_key = secrets.token_urlsafe(32)
    max_hours = cfg.max_hours if args.max_hours is None else args.max_hours
    idle_minutes = cfg.idle_minutes if args.idle is None else args.idle
    deadline = time.time() + max_hours * 3600 if max_hours > 0 else 0
    clouds = [cfg.cloud] + (["SECURE"] if args.secure and cfg.cloud != "SECURE" else [])
    budget = budget_plan(rp, cfg, profile, candidates, clouds, args.budget) if args.budget else None
    if budget:
        candidates = [c for c in candidates if any((c.gpu, cl) in budget for cl in clouds)]

    def guard_deadline(gpu: str, cloud: str) -> float:
        """max_hours, or sooner when --budget runs out first on this GPU."""
        if not budget or (gpu, cloud) not in budget:
            return deadline
        end = time.time() + budget[(gpu, cloud)]["hours"] * 3600
        return min(deadline, end) if deadline else end
    idle_seconds = max(0, idle_minutes) * 60
    # The balance before the pod exists: `ronin down` subtracts the balance then, which
    # gives the real cost at once (per-pod billing is published about an hour later).
    try:
        balance_at_up = rp.account()["balance"]
    except ProviderError:
        balance_at_up = None

    waiting_since = time.time()

    def heartbeat() -> None:
        # visible to `ronin status` / `down` / another `up` in other terminals; re-read
        # so a long wait never writes back an old copy of the state
        fresh = load_state()
        fresh["waiting"] = {"profile": profile.name, "clouds": clouds,
                            "since": waiting_since, "heartbeat": time.time()}
        save_state(fresh)

    build = make_body(provider)
    deadlines = {}                  # (gpu, cloud) -> the deadline its pod body carries

    def body(gpu: str, cloud: str) -> dict:
        deadlines[(gpu, cloud)] = guard_deadline(gpu, cloud)
        return build(cfg, profile, gpu, volume_id, api_key, deadlines[(gpu, cloud)],
                     idle_seconds, cloud)

    try:
        pod, candidate, cloud = create_with_stock(
            rp, cfg, candidates, clouds, body,
            max_wait=(args.max_wait or 0) * 60, on_wait=heartbeat,
            poll=max(MIN_STOCK_POLL, args.poll) if args.poll else STOCK_POLL)
    finally:
        state = load_state()
        if state.pop("waiting", None):
            save_state(state)

    deadline = deadlines.get((candidate.gpu, cloud), deadline)
    state.update(pod_id=pod["id"], profile=profile.name, api_key=api_key, model=profile.model,
                 deadline=deadline, idle_minutes=idle_seconds / 60,
                 created_at=time.time(), balance_at_up=balance_at_up, provider=provider)
    remember_pod(state, pod, profile.name, provider)
    save_state(state)
    ui.ok(f"Pod {pod['id']} created {ui.dim('·')} {pod.get('gpu', {}).get('id', '?')} "
          f"{ui.dim('·')} {hourly_text(pod)} {ui.dim('· ' + cloud.lower() + ' cloud')}")
    if pod.get("where"):
        ui.step(f"Host in {pod['where']}")
    download = download_text(pod, profile)
    if download:
        ui.step(download)
    if cloud != cfg.cloud:
        ui.warn(f"No {cfg.cloud.lower()} stock: this pod is on {cloud.lower()} cloud, "
                "which costs more per hour (allowed by --secure)")
    ui.hint("Live detail: `ronin logs` in another terminal")
    if candidate.tight:
        ui.warn(f"Only ~{candidate.spare_gib:.0f} GiB is left for the KV cache on this GPU")
        ui.hint("If vLLM runs out of memory: lower --max-model-len or add --kv-cache-dtype fp8.")
    plan = (budget or {}).get((candidate.gpu, cloud))
    if plan:
        rate = f"at up to ${plan['hourly']:.2f}/h"
        if plan["download"] >= 0.005:
            rate += f" + ${plan['download']:.2f} download"
        ui.step(f"Budget ${args.budget:.2f}: ≈ {ui.duration(plan['hours'] * 3600)} {ui.dim(rate)}")
    guard = guard_text(state)
    if guard:
        ui.step(f"Guard: {guard}")
    else:
        ui.warn("No guard: the pod bills until `ronin down` (max_hours / idle_minutes are 0)")
    print()

    url = None                      # known once the pod runs (vast.ai publishes the port then)
    logs = LogStream(rp, pod["id"], tail=5000, reconnect=True)
    recent: deque[str] = deque(maxlen=300)
    starts, fatal_at = 0, None
    running, last_poll = False, 0.0
    snapshot, last_snapshot = None, 0.0
    progress = ui.Progress(f"Waiting for {rp.label} to assign the machine")

    def no_gpu(line: str) -> None:
        if any(m in line for m in NO_GPU_MARKERS):
            progress.stop()
            abort_up(rp, pod["id"], recent, keep=args.keep_on_failure, prefer_reason=True,
                     reason=f"the host's GPU is not usable ({line.strip()[-80:]}). "
                            "A broken host, not your profile: `ronin up` again gets another one")

    try:
        # Short timeouts: on Windows Ctrl+C waits for a blocking socket read to finish.
        while not (running and url and healthy(url, timeout=3)):
            now = time.monotonic()
            if (not running or not url) and now - last_poll >= 5:
                last_poll = now
                try:
                    fresh = rp.get_pod(pod["id"], timeout=10)
                except ProviderError as e:
                    if not e.transient:
                        raise
                    fresh = None  # network blip: keep waiting, check again later
                status = fresh["status"] if fresh else None
                if status == "RUNNING":
                    running = True
                    if not url:
                        url = endpoint(fresh, cfg.port)
                        if url:
                            record_url(pod["id"], url)
                elif status in DEAD_STATES:
                    progress.stop()
                    abort_up(rp, pod["id"], recent, keep=args.keep_on_failure,
                             reason=f"the pod ended in state {status}")
                elif progress.elapsed > POD_READY_TIMEOUT[provider]:
                    raise ConfigError("The pod is taking too long to come up\n"
                                      "Check `ronin status`; `ronin down` to delete it.")
            for line in logs.drain():
                recent.append(line)
                if line.startswith(CONTAINER_START):
                    starts += 1
                if fatal_at is None and any(m in line for m in FATAL_MARKERS):
                    fatal_at = time.monotonic()
                no_gpu(line)
                for marker, phase in PHASES:
                    if marker in line:
                        progress.phase(phase)
            # RunPod's long-lived stream can bring the container's lines minutes late; a fresh
            # snapshot every SNAPSHOT_EVERY s catches a dead GPU or a fatal error early.
            # Only those markers are read from it: it repeats lines the stream already gave.
            # (vast.ai's logs are polled, always fresh: no snapshots.)
            if snapshot and snapshot.done:
                for line in snapshot.lines:
                    if line not in recent:
                        recent.append(line)
                    if fatal_at is None and any(m in line for m in FATAL_MARKERS):
                        fatal_at = time.monotonic()
                    no_gpu(line)
                snapshot = None
            if running and snapshot is None and provider == "runpod" \
                    and now - last_snapshot >= SNAPSHOT_EVERY:
                last_snapshot = now
                snapshot = LogSnapshot(rp, pod["id"])
            # the traceback with the cause arrives after the marker: allow some grace
            if starts > 1 or (fatal_at and time.monotonic() - fatal_at > FATAL_GRACE):
                progress.stop()
                abort_up(rp, pod["id"], recent, keep=args.keep_on_failure)
            if progress.elapsed > MODEL_READY_TIMEOUT:
                raise ConfigError("The model server did not respond in time\n"
                                  "The pod is still up: `ronin logs` to investigate, `ronin down` to stop it.")
            progress.tick()
            time.sleep(1)
        if url.startswith("http://"):
            # vast.ai: the guard opens an HTTPS tunnel; use it so the key is not sent in clear
            progress.phase("Opening the HTTPS tunnel")
            secure = wait_for_tunnel(logs, recent, progress)
            if secure:
                url = secure
                record_url(pod["id"], url)
    finally:
        progress.stop()

    ui.ok(f"Model server ready in {ui.duration(progress.elapsed)}")
    if url.startswith("http://"):
        ui.warn("Plain HTTP: the API key and your prompts travel unencrypted to the host")
        ui.hint("The HTTPS tunnel did not come up (`ronin logs` shows why). "
                "`ronin down` if that is not acceptable.")
    print()
    mascot.show("ready", [
        "Ready. The katana is sharp.",
        f"Model: {profile.model}",
        f"Run `ronin {cfg.client}` to begin.",
    ])


def record_url(pod_id: str, url: str) -> None:
    """Save the pod's URL, unless `ronin down` (and maybe a new `up`) ran meanwhile."""
    state = load_state()
    if state.get("pod_id") == pod_id:
        state["url"] = url
        save_state(state)


def tunnel_url(lines) -> str | None:
    for line in reversed(lines):
        match = TUNNEL_LINE.search(line)
        if match:
            return match.group(1)
    return None


def wait_for_tunnel(logs: LogStream, recent, progress) -> str | None:
    """The guard's HTTPS tunnel URL once it answers, or None after TUNNEL_WAIT."""
    started = time.monotonic()
    while time.monotonic() - started < TUNNEL_WAIT:
        recent.extend(logs.drain())
        url = tunnel_url(recent)
        if url and healthy(url, timeout=5):
            return url
        progress.tick()
        time.sleep(1)
    return None


STOCK_POLL = 30                  # seconds between stock checks while waiting for a GPU
MIN_STOCK_POLL = 5               # floor for --poll, to avoid hammering the provider's API


def is_no_stock(e: ProviderError) -> bool:
    """The provider's own reading: out of stock (worth waiting) vs. a refused request."""
    return bool(getattr(e, "no_stock", False))


def waiting_for_stock(state: dict) -> dict | None:
    """The `ronin up` (maybe in another terminal) waiting for stock, if still alive: it
    refreshes a heartbeat every round, so a stale one means that process is gone."""
    w = state.get("waiting")
    if w and time.time() - w.get("heartbeat", 0) <= 2 * STOCK_POLL + 15:
        return w
    return None


def waiting_text(w: dict) -> str:
    return (f"`ronin up -p {w['profile']}` is waiting for {'/'.join(c.lower() for c in w['clouds'])} "
            f"stock ({ui.duration(time.time() - w['since'])} so far) · nothing is billed yet")


def create_with_stock(rp, cfg: Config, candidates, clouds: list[str], make_body,
                      max_wait: float = 0, sleep=time.sleep, on_wait=None,
                      poll: int = STOCK_POLL):
    """Create the pod on the first cloud × GPU with stock, in order. When every attempt
    fails for lack of stock, wait and try again -- nothing is billed while waiting --
    until `max_wait` seconds have passed (0 = until Ctrl+C). Returns (pod, candidate, cloud).

    Errors that are not about stock (an invalid request) are shown once and that GPU is
    still retried with the rest, unless no attempt failed for stock: then waiting is pointless."""
    where = rp.label if getattr(rp, "name", None) == "vast" else (cfg.datacenter or "any datacenter")
    started, shown, progress = time.monotonic(), set(), None
    try:
        while True:
            errors = []
            for cloud in clouds:
                for candidate in candidates:
                    try:
                        pod = rp.create_pod(make_body(candidate.gpu, cloud))
                        return pod, candidate, cloud
                    except ProviderError as e:
                        if e.status in (401, 403):
                            raise
                        errors.append(e)
                        # the provider's own reason: "no stock" and "invalid request" look different
                        key = (cloud, candidate.gpu, e.detail)
                        if key not in shown and not is_no_stock(e):
                            shown.add(key)
                            if progress:
                                progress.stop()
                            ui.warn(f"{candidate.gpu} ({cloud.lower()}) in {where}: {e.detail}")
            # "no stock" may come back as a 5xx: it is not the provider being down
            name = errors[0].provider
            if all(e.transient and not is_no_stock(e) for e in errors):
                raise ConfigError(f"Could not create the pod: {name} is not responding\n"
                                  "Try again in a moment.")
            if not any(is_no_stock(e) for e in errors):
                raise ConfigError("Could not create a pod with any GPU in the profile\n"
                                  f"{name} refused the request (see above); waiting would not help.")
            waited = time.monotonic() - started
            if max_wait and waited >= max_wait:
                raise ConfigError(f"No stock for profile's GPUs after {ui.duration(waited)}\n"
                                  "`ronin gpus` shows current stock; add fallbacks to the profile's "
                                  "`gpus`, or allow secure cloud with `--secure`.")
            if progress is None:
                gpus = ", ".join(c.gpu for c in candidates)
                ui.warn(f"No {'/'.join(c.lower() for c in clouds)} stock right now for: {gpus}")
                hint = "Ctrl+C to cancel"
                if "SECURE" not in clouds:
                    hint += " · `--secure` also tries secure cloud (pricier)"
                ui.hint(f"Nothing is billed while waiting · {hint}")
                progress = ui.Progress(f"Waiting for stock (checking every {poll} s)")
            if on_wait:
                on_wait()
            for _ in range(poll):
                progress.tick()
                sleep(1)
    finally:
        if progress:
            progress.stop()


def abort_up(rp, pod_id: str, recent, keep: bool, reason: str | None = None,
             prefer_reason: bool = False) -> None:
    """The model server will not start: save the log, delete the pod (unless keep) and raise."""
    failure_log = config.project.state_dir / "last-failure.log"
    config.project.state_dir.mkdir(exist_ok=True)
    failure_log.write_text("\n".join(recent) + "\n", encoding="utf-8")
    if prefer_reason and reason:
        cause = reason
    else:
        cause = root_cause(recent) or reason or "cause not identified"
    mascot.show("fail", ["The path has twisted.", "The model server could not start."])
    print()
    msg = [f"The model server could not start: {cause}",
           f"Log saved to {failure_log}"]
    if keep:
        msg.append("The pod is still up (--keep-on-failure): `ronin logs` to investigate, `ronin down` when done.")
    else:
        terminate_own(rp, pod_id)
        msg.append(f"Pod {pod_id} deleted so the GPU stops billing.")
    raise ConfigError("\n".join(msg))


def hourly_text(pod: dict) -> str:
    """'$1.72/h (GPU $1.69 + disk $0.03)': what the pod bills per hour, all in."""
    cost, gpu = pod.get("cost"), pod.get("gpu_cost")
    if not isinstance(cost, (int, float)):
        return "$?/h"
    text = f"${cost:.2f}/h"
    if isinstance(gpu, (int, float)) and cost - gpu >= 0.005:
        text += ui.dim(f" (GPU ${gpu:.2f} + disk ${cost - gpu:.2f})")
    return text


def download_text(pod: dict, profile) -> str | None:
    """What the host bills once for the weight download (vast.ai), or None if it is free."""
    per_gb = pod.get("download_cost_per_gb") or 0
    if per_gb <= 0:
        return None
    try:
        gb = fit.download_gb(profile)
    except LookupError:
        return f"Model download billed by the host at ${per_gb * 1000:.2f}/TB (size unknown)"
    return (f"Model download ≈ ${gb * per_gb:.2f} once "
            f"{ui.dim(f'· {gb:.0f} GB at ${per_gb * 1000:.2f}/TB, billed by the host')}")


MIN_BUDGET_HOURS = 0.25        # a --budget that buys less than this on a GPU skips the GPU


def budget_plan(rp, cfg: Config, profile, candidates, clouds: list[str],
                budget: float) -> dict[tuple[str, str], dict]:
    """--budget: how long each GPU × cloud can run for `budget` dollars, priced now with
    the disk and (vast.ai) the weight download -- and on vast.ai at the dearest offer
    `up` may fall back to, so the budget holds whichever it rents. {(gpu, cloud):
    {'hours', 'hourly', 'download'}}; GPUs without a price, or that `budget` cannot
    keep up for MIN_BUDGET_HOURS, are left out (ConfigError if none is left)."""
    if budget <= 0:
        raise ConfigError("--budget must be more than $0")
    quote = vast_quotes if rp.name == "vast" else runpod_quotes
    try:
        gb = 0.0 if rp.name == "runpod" else fit.download_gb(profile)
    except LookupError as e:
        ui.warn(f"Weight download size unknown ({e}): the budget leaves the download out")
        gb = 0.0
    plan, skipped = {}, []
    for cloud in clouds:
        for q in quote(rp, cfg, [profile], cloud)[profile.name]:
            if not q.gpu_h:
                skipped.append(f"{q.gpu} ({cloud.lower()}): no price")
                continue
            hourly = max(q.hourly, q.hourly_max)
            download = gb * max(q.per_gb, q.per_gb_max)
            hours = (budget - download) / hourly
            if hours < MIN_BUDGET_HOURS:
                skipped.append(f"{q.gpu} ({cloud.lower()}): ${budget:.2f} lasts "
                               f"{ui.duration(max(0, hours) * 3600)} at ${hourly:.2f}/h")
                continue
            plan[(q.gpu, cloud)] = {"hours": hours, "hourly": hourly, "download": download}
    wanted = {c.gpu for c in candidates}
    if not any(gpu in wanted for gpu, _ in plan):
        raise ConfigError(f"--budget ${budget:.2f} is not enough for any GPU of the profile\n"
                          + "\n".join(skipped))
    for line in skipped:
        ui.warn(f"Skipped by the budget: {line}")
    return plan


def estimated_cost(state: dict) -> float | None:
    """$/h of the pod in the history × time since its creation (None if unknown)."""
    entry = next((h for h in reversed(state.get("history", []))
                  if h.get("pod_id") == state.get("pod_id")), None)
    rate, since = (entry or {}).get("cost_per_hour"), state.get("created_at")
    if not isinstance(rate, (int, float)) or not since:
        return None
    return rate * (time.time() - since) / 3600


def cmd_down(args, cfg: Config) -> None:
    state = load_state()
    if not state.get("pod_id"):
        waiting = waiting_for_stock(state)
        if waiting:
            ui.warn("No pod yet, but " + waiting_text(waiting))
            ui.hint("Ctrl+C it in its terminal to cancel; it creates the pod as soon as there is stock.")
            return
        mascot.show("rest", ["No pod registered.", "The ronin already rests."])
        return
    rp = client(pod_provider(state))
    try:
        session = session_cost(rp.get_pod(state["pod_id"]), since=state.get("created_at"))
    except ProviderError:
        session = None
    before = dict(state)            # terminate() clears the pod keys, balance included
    pod_id = terminate(rp, state)
    detail = (f"The weights remain on the network volume (≈ {monthly(cfg.volume_size_gb)})."
              if cfg.storage == "volume" and rp.name == "runpod"
              else "The downloaded weights go with it.")
    lines = [f"Pod {pod_id} deleted.", detail,
             *down_cost_lines(session, real_session_cost(rp, before), estimated_cost(before), rp.label)]
    lines.append("The ronin sheathes the katana.")
    mascot.show("down", lines)


def down_cost_lines(session: str | None, real: float | None, estimate: float | None,
                    label: str) -> list[str]:
    """The cost lines of `ronin down`. The provider debits the balance in batches (RunPod:
    every ~15 min, the last one after the pod is gone), so right after a delete the balance
    drop trails the time billed -- $0.00 for a short session. It is shown as the cost only
    once it has caught up with the estimate; until then the estimate is the figure."""
    lines = []
    if session:
        lines.append(ui.dim(f"Session: {session}"))
    elif estimate is not None:
        lines.append(ui.dim(f"Session: ≈ ${estimate:.2f} (estimate)"))
    if real is None:
        return lines
    if real >= 0.005 and (estimate is None or real >= estimate * 0.9):
        lines.append(f"Real cost: ${real:.2f} {ui.dim('(account balance since `ronin up`)')}")
    elif estimate is not None:
        debited = f"has debited ${real:.2f} of it so far" if real >= 0.005 else "has not debited it yet"
        lines.append(ui.dim(f"{label} {debited} (it bills in batches); "
                            "`ronin cost` has the final figure in about an hour."))
    return lines


STATUS_READY, STATUS_NOT_READY, STATUS_NO_POD = 0, 2, 3   # `ronin status` exit codes


def cmd_status(args, cfg: Config) -> int:
    state = load_state()
    info = status_info(cfg, state)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        show_status(cfg, state, info)
    if not info["pod"]:
        # waiting for stock is "coming, not ready", not "nothing there"
        return STATUS_NOT_READY if info.get("waiting") else STATUS_NO_POD
    return STATUS_READY if info["ready"] else STATUS_NOT_READY


def status_info(cfg: Config, state: dict) -> dict:
    """Everything `ronin status` knows, as plain data (also its --json output)."""
    provider = pod_provider(state) if state.get("pod_id") else cfg.provider
    info = {"pod": None, "ready": False, "storage": cfg.storage, "provider": provider,
            "volume_id": state.get("volume_id") if cfg.storage == "volume" else None,
            "account": None}
    waiting = waiting_for_stock(state)
    info["waiting"] = ({"profile": waiting["profile"], "clouds": waiting["clouds"],
                        "since": waiting["since"]} if waiting else None)
    try:
        rp = client(provider)
        info["account"] = rp.account()      # nice to have: never a reason for status to fail
    except (ConfigError, ProviderError):
        rp = None
    info["others"] = [{"id": p["id"], "name": p.get("name"), "status": p.get("status"),
                       "cost_per_hour": p.get("cost"), "provider": other.name}
                      for other, p in orphans(state)]
    if not state.get("pod_id"):
        return info
    rp = rp or client(provider)
    try:
        pod = rp.get_pod(state["pod_id"])
    except ProviderError as e:
        if e.status != 404:
            raise
        info["pod"] = {"id": state["pod_id"], "status": "GONE"}
        return info
    uptime = (pod.get("runtime") or {}).get("uptime")
    cost = pod.get("cost")
    # billed time runs from creation (image pull and download included), not container uptime
    billed = time.time() - state["created_at"] if state.get("created_at") else uptime
    url = state.get("url") or endpoint(pod, cfg.port)
    info["ready"] = bool(pod["status"] == "RUNNING" and url and healthy(url, timeout=5))
    info["pod"] = {
        "id": pod["id"], "status": pod["status"], "url": url,
        "profile": state.get("profile"), "model": state.get("model"),
        "gpu": pod.get("gpu", {}).get("id"), "cost_per_hour": cost,
        "uptime_seconds": uptime,
        "created_at": state.get("created_at"),
        "session_cost_estimate": round(cost * billed / 3600, 2)
        if isinstance(cost, (int, float)) and billed is not None else None,
        "session_cost_real": round(real, 2)
        if (real := real_session_cost(rp, state)) is not None else None,
        # kept for scripts that read it; the estimate
        "session_cost": round(cost * billed / 3600, 2)
        if isinstance(cost, (int, float)) and billed is not None else None,
        "guard": {"deadline": state.get("deadline") or None,
                  "idle_minutes": state.get("idle_minutes") or None},
    }
    return info


def show_status(cfg: Config, state: dict, info: dict) -> None:
    storage = (f"volume {state['volume_id']} ({cfg.datacenter}) "
               f"{ui.dim('· ' + str(cfg.volume_size_gb) + ' GB · ≈ ' + monthly(cfg.volume_size_gb))}"
               if cfg.storage == "volume" and state.get("volume_id") else cfg.storage)
    pod = info["pod"]
    account = info.get("account")
    account_line = (ui.field("Account", f"${account['balance']:.2f} left {ui.dim('·')} "
                                        f"spending ${account['spend_per_hour']:.2f}/h now")
                    if account else None)
    warnings = [*runway_warning(account, state if pod else {}), *others_lines(info.get("others"))]
    if not pod:
        waiting = info.get("waiting")
        lines = [ui.field("Pod", ui.yellow("waiting for stock") if waiting else ui.dim("none")),
                 ui.field("Storage", storage)]
        if account_line:
            lines.append(account_line)
        lines += warnings
        if waiting:
            mascot.show("wait", [*lines, "", waiting_text(waiting),
                                 "Ctrl+C it in its terminal to cancel."], top=True)
        else:
            mascot.show("rest", [*lines, "", "The ronin rests. `ronin up` to wake it."], top=True)
        return
    label = PROVIDERS[info.get("provider") or "runpod"][0].label
    if pod["status"] == "GONE":
        mascot.show("rest", [
            ui.field("Pod", f"{pod['id']} {ui.yellow('no longer exists on ' + label)}"),
            "",
            "`ronin down` to clear the local state.",
        ], top=True)
        return
    ok = info["ready"]
    status_color = ui.green if pod["status"] == "RUNNING" else ui.yellow
    cost = "?" if pod["cost_per_hour"] is None else f"{pod['cost_per_hour']:.2f}"
    lines = [
        ui.field("Pod", f"{pod['id']} {status_color(pod['status'])}"),
        ui.field("Profile", f"{pod['profile']} {ui.dim('· ' + str(pod['model']))}"),
        ui.field("GPU", f"{pod['gpu'] or '?'} {ui.dim('·')} ${cost}/h {ui.dim('· ' + label)}"),
    ]
    uptime = pod["uptime_seconds"]
    session = session_cost({"runtime": {"uptime": uptime}, "cost": pod["cost_per_hour"]},
                           since=pod.get("created_at"))
    if session:
        if pod.get("session_cost_real") is not None:
            # the balance is debited in batches, so this trails the estimate
            session += f" {ui.dim('·')} debited so far ${pod['session_cost_real']:.2f}"
        lines.append(ui.field("Session", session))
    guard = guard_text(state)
    lines.append(ui.field("Guard", guard if guard else ui.yellow("none · bills until `ronin down`")))
    lines += [
        ui.field("URL", ui.dim(pod["url"] or "not published yet")
                 + (ui.yellow(" · plain HTTP") if (pod["url"] or "").startswith("http://") else "")),
        ui.field("Server", ui.green("ready") if ok else ui.yellow("not responding yet")),
        ui.field("Storage", storage),
    ]
    if account_line:
        lines.append(account_line)
    lines += warnings
    if uptime and uptime > WARN_AFTER_HOURS * 3600:
        lines += ["", ui.yellow(f"! Running for {ui.duration(uptime)} and still billing — "
                                "`ronin down` if you are done.")]
    mascot.show("ready" if ok else "wait", lines, top=True)


RUNWAY_WARN_HOURS = 2         # warn when the balance lasts less than this at the current spend


def runway_hours(balance: float, per_hour: float) -> float | None:
    """Hours the balance lasts at `per_hour` (None when nothing is billing)."""
    return max(0.0, balance) / per_hour if per_hour > 0 else None


def runway_warning(account: dict | None, state: dict) -> list[str]:
    """A warning when the balance runs out soon, or before the guard's deadline: the
    provider then stops the pod mid-session (and its disk may go on billing)."""
    if not account:
        return []
    hours = runway_hours(account["balance"], account["spend_per_hour"])
    if hours is None:
        return []
    deadline = state.get("deadline") or 0
    left = (deadline - time.time()) / 3600 if deadline else None
    if hours >= RUNWAY_WARN_HOURS and (left is None or hours >= left):
        return []
    return ["", ui.yellow(f"! The balance lasts ≈ {ui.duration(hours * 3600)} at this spend"
                          + (f", before the guard's deadline (in {ui.duration(left * 3600)})"
                             if left is not None and hours < left else "")
                          + ": add credit, or the provider stops the pod.")]


def others_lines(others: list[dict] | None) -> list[str]:
    """Pods named ronin-* this project has not registered: they bill unseen."""
    if not others:
        return []
    lines = ["", ui.yellow(f"! {len(others)} other ronin pod{'s' if len(others) > 1 else ''} "
                           "on the account, not registered here:")]
    for o in others:
        label = PROVIDERS[o["provider"]][0].label
        cost = o.get("cost_per_hour")
        rate = f"${cost:.2f}/h" if isinstance(cost, (int, float)) else "$?/h"
        lines.append(f"  {o['id']}  {o.get('name') or '-'}  {o.get('status') or '?'}  "
                     f"{rate}  {ui.dim(label)}")
    lines.append(ui.dim("Another project's, or left over by a crashed `up`: delete it in the "
                        "provider's console if it is not in use."))
    return lines


def billed_by_pod(rows: list[dict]) -> dict[str, dict]:
    """Billing rows summed per pod: {pod_id: {'amount', 'minutes', 'first'}}."""
    pods: dict[str, dict] = {}
    for row in rows:
        p = pods.setdefault(row["podId"], {"amount": 0.0, "minutes": 0.0, "first": row["time"]})
        p["amount"] += row.get("amount") or 0
        p["minutes"] += (row.get("timeBilledMs") or 0) / 60000
        p["first"] = min(p["first"], row["time"])
    return pods


def cmd_cost(args, cfg: Config) -> None:
    """What the provider actually charged, per pod, from its billing API (RunPod), plus
    estimates for the pods its billing does not show."""
    from datetime import datetime, timedelta, timezone
    provider = args.provider or cfg.provider
    rp = client(provider)
    state = load_state()
    since = (datetime.now(timezone.utc) - timedelta(days=args.days - 1)).strftime("%Y-%m-%dT00:00:00Z")
    rows = rp.pod_billing(since, bucket="day")
    pods = billed_by_pod(rows or [])
    history = {h["pod_id"]: h for h in state.get("history", [])
               if (h.get("provider") or "runpod") == provider}
    if rows is None:
        print(f"  {ui.bold(f'Pods ronin created on {rp.label} since {since[:10]}')} "
              f"{ui.dim('(estimates: per-pod billing is not read from ' + rp.label + ')')}\n")
    else:
        print(f"  {ui.bold(f'Billed by {rp.label} since {since[:10]}')} {ui.dim('(UTC · real amounts)')}\n")
    if pods:
        print("  " + ui.dim(f"{'pod':<16}  {'profile':<16}  {'day':<10}  {'billed':>8}  {'amount':>8}"))
        for pod_id, p in sorted(pods.items(), key=lambda kv: kv[1]["first"]):
            # pad before colouring: ANSI codes would count towards the width
            name = history.get(pod_id, {}).get("profile")
            profile = name.ljust(16) if name else ui.dim("-".ljust(16))
            print(f"  {pod_id:<16}  {profile}  {p['first'][:10]:<10}  "
                  f"{ui.duration(p['minutes'] * 60):>8}  {'$' + format(p['amount'], '.2f'):>8}")
        print(f"\n  Total billed: {ui.bold(f'${sum(p['amount'] for p in pods.values()):.2f}')}")
    elif rows is not None:
        ui.step("Nothing billed in this period yet.")
    # pods ronin created that billing does not show yet (it is published ~1 h late)
    cutoff = datetime.fromisoformat(since.replace("Z", "+00:00")).timestamp()
    pending = [h for h in history.values()
               if h.get("created_at", 0) >= cutoff and h["pod_id"] not in pods]
    if rows is None and not pending:
        ui.step("No pods in this period.")
    elif pending and rows is None:
        ui.step(f"{len(pending)} pod(s):")
    elif pending:
        print()
        ui.warn(f"{len(pending)} pod(s) not in {rp.label}'s billing yet (it is published about an hour late):")
        for h in pending:
            end = h.get("deleted_at") or time.time()
            secs = end - h["created_at"]
            rate = h.get("cost_per_hour")
            est = f"≈ ${rate * secs / 3600:.2f} (estimate)" if isinstance(rate, (int, float)) else "?"
            running = "" if h.get("deleted_at") else ui.yellow(" · still registered")
            print(f"    {h['pod_id']:<16}  {h['profile']:<16}  {ui.duration(secs):>8}  {est}{running}")
    try:
        account = rp.account()
        print()
        ui.step(f"Account: ${account['balance']:.2f} left · spending ${account['spend_per_hour']:.2f}/h now")
    except ProviderError as e:
        ui.warn(f"Account balance unavailable: {e.detail}")


def cmd_logs(args, cfg: Config) -> None:
    state = load_state()
    if not state.get("pod_id"):
        ui.warn("No pod registered.")
        ui.hint("`ronin up` to start one.")
        return
    if args.no_follow:
        ui.step(f"Last {args.tail} log lines of pod {state['pod_id']}")
        print()
        logs = LogStream(client(pod_provider(state)), state["pod_id"], tail=args.tail, reconnect=False)
        # the SSE stream stays open, so stop once the backfill goes quiet
        while True:
            try:
                item = logs.lines.get(timeout=3.0)
            except queue.Empty:
                return
            if item is LogStream.DONE:
                return
            if isinstance(item, Exception):
                raise item
            print(item, flush=True)
    ui.step(f"Logs of pod {state['pod_id']} {ui.dim('· Ctrl+C to exit')}")
    print()
    logs = LogStream(client(pod_provider(state)), state["pod_id"], tail=args.tail)
    while True:
        try:
            item = logs.lines.get(timeout=0.5)  # short timeout: let Ctrl+C through
        except queue.Empty:
            continue
        if item is LogStream.DONE:
            return
        if isinstance(item, Exception):
            raise item
        print(item, flush=True)


def write_opencode_config(cfg: Config, state: dict) -> Path:
    """Write the generated opencode.json into .ronin/ and return its path. The API key
    is NOT written: opencode reads it from the RONIN_API_KEY environment variable."""
    config.project.state_dir.mkdir(exist_ok=True)
    path = config.project.state_dir / "opencode.json"
    path.write_text(json.dumps(opencode_config(cfg, state), indent=2), encoding="utf-8")
    return path


def cmd_env(args, cfg: Config) -> None:
    state = load_state()
    if cfg.client == "claude":
        env = claude_env(cfg, state)
        if args.shell == "powershell":
            print("Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue")
        else:
            print("unset ANTHROPIC_API_KEY")
    else:
        path = write_opencode_config(cfg, state)
        env = {"OPENCODE_CONFIG": str(path), OPENCODE_API_KEY_ENV: state["api_key"]}
    for key, value in env.items():
        if args.shell == "powershell":
            print(f"$env:{key} = '{value}'")
        else:
            print(f"export {key}='{value}'")
    usage = "ronin env | Invoke-Expression" if args.shell == "powershell" else 'eval "$(ronin env)"'
    print(f"# Apply to the current shell with: {usage}")


def ensure_pod_for_agent(args, cfg: Config) -> dict:
    """Shared by the launch command: validate the pod (optionally bringing one up with
    --up) and return the current state. Raises if there is no usable pod."""
    state = load_state()
    if args.profile and not args.up:
        ui.warn(f"--profile is only used with --up; using the active pod "
                f"({state.get('profile') or 'none'})")
    if state.get("pod_id"):
        problem = pod_problem(cfg, state)
        if problem and args.up and problem.startswith("gone"):
            state = load_state()            # pod_problem cleared it: bring a new one up
        elif problem:
            hint = f"`ronin {cfg.client} --up` to start a new one" if problem.startswith("gone") \
                else "`ronin status` · `ronin logs`"
            raise ConfigError(f"The pod is {problem}\n{hint}")
    if args.up and not state.get("pod_id"):
        cmd_up(argparse.Namespace(profile=args.profile, keep_on_failure=False,
                                  max_hours=None, idle=None, skip_memory_check=False,
                                  secure=args.secure, max_wait=args.max_wait,
                                  poll=args.poll, provider=args.provider), cfg)
        print()
    return load_state()


def cmd_agent(args, cfg: Config) -> None:
    """Open the configured coding agent (opencode or Claude Code) against the pod."""
    if cfg.client == "claude":
        exe = shutil.which("claude")
        if not exe:
            raise ConfigError("Could not find the `claude` executable on PATH\n"
                              "Install Claude Code: https://claude.com/claude-code")
    else:
        exe = shutil.which(cfg.opencode_bin)
        if not exe:
            raise ConfigError(f"Could not find the `{cfg.opencode_bin}` executable on PATH\n"
                              "Install opencode: https://opencode.ai")
    state = ensure_pod_for_agent(args, cfg)
    if cfg.client == "claude":
        env = {**os.environ, **claude_env(cfg, state)}
        env.pop("ANTHROPIC_API_KEY", None)
        cmd = [exe, *args.agent_args]
    else:
        path = write_opencode_config(cfg, state)
        env = {**os.environ, "OPENCODE_CONFIG": str(path),
               OPENCODE_API_KEY_ENV: state["api_key"]}
        cmd = [exe, *opencode_standalone(args.agent_args)]
    sys.exit(subprocess.call(cmd, env=env))


def opencode_standalone(agent_args: list[str]) -> list[str]:
    """opencode's shared background service does not pick up OPENCODE_CONFIG, so ronin
    forces a private in-process server with --standalone (the only reliable way to load
    ronin's provider). The flag goes after the subcommand when there is one (e.g.
    `opencode run --standalone ...`) and leads for the bare TUI (`opencode --standalone`)."""
    if agent_args and not agent_args[0].startswith("-"):     # a subcommand like `run`
        return [agent_args[0], "--standalone", *agent_args[1:]]
    return ["--standalone", *agent_args]


# Commands whose output is not framed with blank lines: `env` is eval'd by the shell
# and the agent (opencode/claude) opens a full-screen interface.
UNPADDED = ("env", "opencode", "claude")


# Commands that can stop half-way with a pod still up: remind that it bills.
BILLING_HINT = ("up", "watchdog", "opencode", "claude")


# Commands that load (or create) the config themselves, to report problems with it.
OWN_CONFIG = ("init", "doctor")


# Commands that must work even with a broken ronin.toml: a typo must never keep you
# from seeing or deleting a pod that bills. They fall back to the default settings.
LENIENT = ("down", "status", "logs")


def command_config(command: str) -> Config | None:
    if command in OWN_CONFIG:
        return None
    try:
        return load_config()
    except ConfigError as e:
        if command not in LENIENT:
            raise
        ui.warn(f"ronin.toml has problems; using default settings for `{command}`")
        for line in str(e).splitlines()[1:4]:
            ui.hint(line)
        ui.hint("`ronin doctor` lists every problem")
        print()
        return config.fallback_config()


def save_traceback() -> Path | None:
    try:
        import traceback
        config.project.state_dir.mkdir(exist_ok=True)
        path = config.project.state_dir / "last-error.log"
        path.write_text(traceback.format_exc(), encoding="utf-8")
        return path
    except OSError:
        return None


def _pod_registered() -> bool:
    try:
        return bool(load_state().get("pod_id"))
    except ConfigError:
        return False


GLOBAL_VALUED = ("--config",)               # global options that take a value


LAUNCH_COMMANDS = ("opencode", "claude")    # verbs that forward trailing args to the agent


AGENT_FLAGS = ("--up", "--secure", "-h", "--help")   # ronin's own launch options without a value


AGENT_VALUED = ("--profile", "--max-wait", "--poll", "--provider")  # ... and with one


def add_provider_option(p, what: str) -> None:
    p.add_argument("--provider", choices=config.PROVIDERS,
                   help=f"{what} (default: provider in ronin.toml)")


def add_stock_options(p) -> None:
    p.add_argument("--secure", action="store_true",
                   help="if the configured cloud (community) has no stock, use secure cloud "
                        "(pricier) instead of waiting")
    p.add_argument("--max-wait", type=float, metavar="MINUTES",
                   help="give up after waiting this long for stock (default: wait until Ctrl+C)")
    p.add_argument("--poll", type=int, metavar="SECONDS",
                   help=f"seconds between stock attempts while waiting (default {STOCK_POLL}; "
                        f"minimum {MIN_STOCK_POLL})")


def split_agent_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split `ronin opencode`/`ronin claude` argv into ronin's part and the part passed
    to the agent.

    argparse cannot pass through unknown options (`--resume`, `-p`), so ronin's own
    options are taken only while they lead; the first other token, or `--`, starts
    the agent's arguments.
    """
    i = 0
    while i < len(argv) and argv[i].startswith("-"):    # global options
        i += 2 if argv[i] in GLOBAL_VALUED else 1
    if i >= len(argv) or argv[i] not in LAUNCH_COMMANDS:
        return argv, []
    j = i + 1
    while j < len(argv):
        token = argv[j]
        if token == "--":
            return argv[:j], argv[j + 1:]
        if token in AGENT_FLAGS or any(token.startswith(v + "=") for v in AGENT_VALUED):
            j += 1
        elif token in AGENT_VALUED:
            j += 2
        else:
            break
    return argv[:j], argv[j:]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ronin", description=__doc__)
    parser.add_argument("--version", action="version", version=f"ronin {__version__}")
    parser.add_argument("--config", metavar="PATH",
                        help="ronin.toml to use (default: RONIN_CONFIG, else the nearest one "
                             "from the current directory up)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="set up a project here: ronin.toml, .env and checks")
    p.set_defaults(func=cmd_init)
    sub.add_parser("doctor", help="check the setup: config, API key, agent, stock").set_defaults(func=cmd_doctor)

    p = sub.add_parser("volume", help="list, create or delete RunPod network volumes")
    vsub = p.add_subparsers(dest="action")
    vsub.add_parser("ls", help="list volumes with size and monthly cost")
    vsub.add_parser("create", help="create the volume configured in ronin.toml")
    v = vsub.add_parser("rm", help="delete a volume and the models on it")
    v.add_argument("id")
    v.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=cmd_volume, action="ls")
    sub.add_parser("profiles", help="list the configured profiles").set_defaults(func=cmd_profiles)

    p = sub.add_parser("gpus", help="list GPUs, price and datacenters with stock")
    p.add_argument("--min-vram", type=int, default=24, help="minimum VRAM in GB (default 24)")
    p.add_argument("--secure", action="store_true",
                   help="list secure cloud instead of the configured cloud")
    add_provider_option(p, "list this provider's GPUs")
    p.set_defaults(func=cmd_gpus)

    p = sub.add_parser("prices", help="what each profile costs now: GPU, disk, download and total")
    p.add_argument("profiles", nargs="*", metavar="PROFILE", help="profiles to price (default: all)")
    p.add_argument("--hours", type=float, default=1,
                   help="session length for the total (default 1)")
    p.add_argument("--secure", action="store_true",
                   help="price secure cloud instead of the configured cloud")
    add_provider_option(p, "price on this provider")
    p.set_defaults(func=cmd_prices)

    p = sub.add_parser("up", help="bring the pod up and wait until the model server is ready")
    p.add_argument("-p", "--profile", help="profile from ronin.toml (default: the one in [defaults])")
    p.add_argument("--keep-on-failure", action="store_true",
                   help="if the server fails to start, do not delete the pod (to investigate with `ronin logs`)")
    p.add_argument("--budget", type=float, metavar="USD",
                   help="dollars this pod may spend: the guard deletes it when they run out "
                        "(GPU + disk, and the weight download on vast.ai, priced now)")
    p.add_argument("--max-hours", type=float,
                   help="the pod deletes itself after this many hours (default: max_hours in ronin.toml; 0 = never)")
    p.add_argument("--idle", type=float, metavar="MINUTES",
                   help="the pod deletes itself after this many minutes without requests "
                        "(default: idle_minutes in ronin.toml; 0 = never)")
    p.add_argument("--skip-memory-check", action="store_true",
                   help="try the profile's GPUs even if the weights look too big for them")
    add_stock_options(p)
    add_provider_option(p, "where to run the pod")
    p.set_defaults(func=cmd_up)

    sub.add_parser("down", help="delete the pod (the volume is kept)").set_defaults(func=cmd_down)
    p = sub.add_parser("cost", help="what the provider billed, per pod, and the account balance")
    p.add_argument("--days", type=int, default=7, help="days back, today included (default 7)")
    add_provider_option(p, "whose billing to read")
    p.set_defaults(func=cmd_cost)
    p = sub.add_parser("status", help="pod state, cost and URL",
                       epilog="Exit code: 0 ready, 2 pod not ready or gone, 3 no pod.")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("watchdog", help="auto-delete the pod after a maximum of billing hours")
    p.add_argument("--max-hours", type=float,
                   help="delete the pod once its total uptime reaches this many hours "
                        "(default: max_hours in ronin.toml)")
    p.add_argument("--poll", type=int, default=WATCHDOG_POLL,
                   help=f"seconds between provider checks (default {WATCHDOG_POLL})")
    p.set_defaults(func=cmd_watchdog)

    p = sub.add_parser("logs", help="stream the pod logs live")
    p.add_argument("-n", "--tail", type=int, default=100)
    p.add_argument("--no-follow", action="store_true", help="print the backlog and exit")
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("env", help="print the variables/config that point the agent at the pod")
    p.add_argument("--shell", choices=["powershell", "bash"],
                   default="powershell" if os.name == "nt" else "bash")
    p.set_defaults(func=cmd_env)

    # `claude` is a hidden alias for muscle memory; both launch the agent set by `client`.
    p = sub.add_parser(
        "opencode", aliases=["claude"], help="open the coding agent connected to the pod",
        usage="ronin opencode [--up] [--profile PROFILE] [--] [agent arguments...]",
        epilog="Launches the agent set by `client` in ronin.toml (opencode by default). "
               "Everything after ronin's own options is passed to the agent untouched, "
               "e.g. `ronin opencode --up` or `ronin opencode -p \"explain this repo\"`.")
    # no -p short form: agents use it for print mode
    p.add_argument("--up", action="store_true", help="bring a pod up first if none is active")
    p.add_argument("--profile", help="profile to use with --up")
    add_stock_options(p)
    add_provider_option(p, "where --up runs the pod")
    p.set_defaults(func=cmd_agent)

    argv = sys.argv[1:] if argv is None else list(argv)
    argv, agent_args = split_agent_args(argv)
    args = parser.parse_args(argv)
    args.agent_args = agent_args
    config.use_project(args.config)
    load_env()
    padded = args.command not in UNPADDED and not getattr(args, "json", False)
    if padded:
        print()
    code = 0
    try:
        code = args.func(args, command_config(args.command)) or 0
    except (ConfigError, ProviderError) as e:
        sys.stdout.flush()
        ui.error(str(e))
        code = 1
        # The error may have stopped a command mid-way while the GPU keeps billing.
        if args.command in BILLING_HINT and _pod_registered():
            ui.hint("A pod is still registered and may be billing: `ronin status` · `ronin down`")
    except (OSError, json.JSONDecodeError) as e:
        sys.stdout.flush()
        ui.error(f"Unexpected error: {e}")
        code = 1
    except KeyboardInterrupt:
        print()
        ui.warn("Interrupted.")
        if args.command in BILLING_HINT and _pod_registered():
            ui.hint("The pod is still up and billing: `ronin status` · `ronin down`")
        code = 130
    except Exception as e:
        # a bug: say so plainly, keep the traceback for the report, and never let it
        # hide a pod that may be billing
        sys.stdout.flush()
        ui.error(f"Unexpected error (a bug in ronin): {type(e).__name__}: {e}")
        log = save_traceback()
        if log:
            ui.hint(f"Traceback saved to {log}")
        if _pod_registered():
            ui.hint("A pod is registered and may be billing: `ronin status` · `ronin down`")
        code = 1
    if padded:
        sys.stderr.flush()
        print()
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
