"""ronin CLI: init, doctor, gpus, profiles, volume, up, down, status, cost, logs,
watchdog, env, opencode."""

from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from . import __version__, config, fit, mascot, ui
from .config import Config, ConfigError, load_config, load_env, load_state, save_state
from .pod import (CONTAINER_START, DEAD_STATES, FATAL_MARKERS, NO_GPU_MARKERS,
                  OPENCODE_API_KEY_ENV, LogSnapshot, LogStream,
                  clear_pod, client, claude_env, ensure_volume, guard_text, healthy,
                  max_model_len, monthly, opencode_config, pod_body, pod_problem, proxy_url,
                  real_session_cost, remember_pod, root_cause, session_cost, terminate,
                  terminate_own)
from .runpod import RunPod, RunPodError
from .watchdog import WATCHDOG_POLL, cmd_watchdog

POD_READY_TIMEOUT = 10 * 60      # waiting for the container to come up


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
    if not os.environ.get("RUNPOD_API_KEY"):
        print()
        ui.warn("RUNPOD_API_KEY is empty")
        ui.hint(f"Paste your key into {project.env} (https://console.runpod.io/user/settings), "
                "then run `ronin doctor`.")
        return
    cfg = load_config()
    if cfg.storage == "volume":
        vol_id = ensure_volume(client(), cfg, load_state())
        ui.ok(f"Network volume ready: {vol_id} ({cfg.datacenter})")
    print()
    ui.hint("Next: `ronin doctor` to check everything, then `ronin up`.")


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

    rp = None
    if not os.environ.get("RUNPOD_API_KEY"):
        bad("RUNPOD_API_KEY is not set", f"Add it to {project.env} (`ronin init` creates the file).")
    else:
        rp = client()
        try:
            gpus = rp.list_gpus(cfg.cloud if cfg else "SECURE")
        except RunPodError as e:
            hint = "Check the key in .env." if e.status in (401, 403) else None
            bad(f"RunPod API: {e.detail}", hint)
            rp = None
        else:
            ui.ok("RunPod API key works")
            if cfg:
                profile = cfg.profile(None)
                stock = {g["id"]: g for g in gpus}
                available = [g for g in profile.gpus
                             if stock.get(g, {}).get("availability") not in (None, "NONE")
                             and (not cfg.datacenter or cfg.datacenter in
                                  [dc["id"] for dc in stock.get(g, {}).get("dataCenters", [])])]
                where = cfg.datacenter or "any datacenter"
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

    print()
    if problems:
        raise ConfigError(f"{problems} problem{'s' if problems > 1 else ''} found")
    ui.ok("All good. `ronin up` to start.")


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


def memory_check(rp: RunPod, cfg: Config, profile) -> list[fit.Fit]:
    """The profile's GPUs that can hold the weights, in order; warns about the rest.
    Stops `ronin up` before any pod is created (and billed) when none can."""
    unchecked = [fit.Fit(g, None, None) for g in profile.gpus]
    if profile.backend != "vllm":
        ui.step(f"Memory check skipped: {profile.backend} / GGUF has no safetensors size")
        return unchecked
    try:
        params = fit.checkpoint_params(profile.model)
        catalog = rp.list_gpus(cfg.cloud)
    except (LookupError, RunPodError) as e:
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
    volumes = client().list_volumes()
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
    vol_id = ensure_volume(client(), cfg, load_state())
    ui.ok(f"Network volume ready: {vol_id} ({cfg.datacenter})")


def volume_rm(args, cfg: Config) -> None:
    rp = client()
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
    cloud = cfg.cloud.lower()
    gpus = [g for g in client().list_gpus(cfg.cloud) if g.get("memory", 0) >= args.min_vram]
    gpus.sort(key=lambda g: (g.get("price", {}).get(cloud) or 0))
    if not gpus:
        ui.warn(f"No GPUs with {args.min_vram} GB or more in {cfg.cloud}.")
        return

    stock_color = {"HIGH": ui.green, "MEDIUM": ui.yellow, "LOW": ui.yellow}
    rows = []
    for g in gpus:
        price = g.get("price", {}).get(cloud)
        dcs = [dc["id"] for dc in g.get("dataCenters", [])]
        rows.append((g["id"], f"{g['memory']} GB", f"${price:.2f}" if price is not None else "-",
                     g.get("availability", "-"), dcs))
    w_id = max(len(r[0]) for r in rows)

    print(f"  {ui.bold(f'GPUs with {args.min_vram} GB or more')} {ui.dim(f'· {cfg.cloud} cloud · price per hour')}\n")
    print("  " + ui.dim(f"{'GPU':<{w_id}}  {'VRAM':>6}  {'$/h':>6}  {'stock':<6}  datacenters"))
    for gpu_id, vram, price, stock, dcs in rows:
        color = stock_color.get(stock, ui.dim)
        dc_text = ", ".join(ui.bold(d) if d == cfg.datacenter else d for d in dcs) or ui.dim("-")
        print(f"  {gpu_id:<{w_id}}  {vram:>6}  {price:>6}  {color(f'{stock:<6}')}  {dc_text}")
    if cfg.datacenter:
        print()
        ui.hint(f"Bold: your datacenter ({cfg.datacenter}).")


def cmd_up(args, cfg: Config) -> None:
    profile = cfg.profile(args.profile)
    rp = client()
    state = load_state()

    waiting = waiting_for_stock(state)
    if waiting:
        # two waiting `up`s could each create a pod, and both would bill
        ui.warn("Another " + waiting_text(waiting))
        ui.hint("Wait for it, or Ctrl+C it in its terminal first.")
        return

    if state.get("pod_id"):
        try:
            pod = rp.get_pod(state["pod_id"])
        except RunPodError as e:
            if e.status != 404:
                raise
            clear_pod(state)
        else:
            if pod["status"] not in DEAD_STATES:
                ui.warn(f"A pod is already active: {pod['id']} ({pod['status']}) "
                        f"· profile {state.get('profile')}")
                ui.hint("`ronin status` to see it · `ronin down` to delete it")
                return
            # A stopped pod still exists (and keeps its disk): delete it, do not orphan it.
            ui.step(f"Deleting the previous pod {pod['id']} ({pod['status']})")
            terminate(rp, state)

    ui.step(f"Profile {ui.bold(profile.name)} {ui.dim('· ' + profile.model)}")
    if args.skip_memory_check:
        candidates = [fit.Fit(g, None, None) for g in profile.gpus]
    else:
        candidates = memory_check(rp, cfg, profile)
    volume_id = ensure_volume(rp, cfg, state) if cfg.storage == "volume" else None
    api_key = secrets.token_urlsafe(32)
    max_hours = cfg.max_hours if args.max_hours is None else args.max_hours
    idle_minutes = cfg.idle_minutes if args.idle is None else args.idle
    deadline = time.time() + max_hours * 3600 if max_hours > 0 else 0
    idle_seconds = max(0, idle_minutes) * 60
    # The balance before the pod exists: `ronin down` subtracts the balance then, which
    # gives the real cost at once (per-pod billing is published about an hour later).
    try:
        balance_at_up = rp.account()["balance"]
    except RunPodError:
        balance_at_up = None

    clouds = [cfg.cloud] + (["SECURE"] if args.secure and cfg.cloud != "SECURE" else [])
    waiting_since = time.time()

    def heartbeat() -> None:
        # visible to `ronin status` / `down` / another `up` in other terminals; re-read
        # so a long wait never writes back an old copy of the state
        fresh = load_state()
        fresh["waiting"] = {"profile": profile.name, "clouds": clouds,
                            "since": waiting_since, "heartbeat": time.time()}
        save_state(fresh)

    try:
        pod, candidate, cloud = create_with_stock(
            rp, cfg, candidates, clouds,
            lambda gpu, cloud: pod_body(cfg, profile, gpu, volume_id, api_key,
                                        deadline, idle_seconds, cloud=cloud),
            max_wait=(args.max_wait or 0) * 60, on_wait=heartbeat)
    finally:
        state = load_state()
        if state.pop("waiting", None):
            save_state(state)

    state.update(pod_id=pod["id"], profile=profile.name, api_key=api_key, model=profile.model,
                 deadline=deadline, idle_minutes=idle_seconds / 60,
                 created_at=time.time(), balance_at_up=balance_at_up)
    remember_pod(state, pod, profile.name)
    save_state(state)
    ui.ok(f"Pod {pod['id']} created {ui.dim('·')} {pod.get('gpu', {}).get('id', '?')} "
          f"{ui.dim('·')} ${pod.get('cost', '?')}/h {ui.dim('· ' + cloud.lower() + ' cloud')}")
    if cloud != cfg.cloud:
        ui.warn(f"No {cfg.cloud.lower()} stock: this pod is on {cloud.lower()} cloud, "
                "which costs more per hour (allowed by --secure)")
    ui.hint("Live detail: `ronin logs` in another terminal")
    if candidate.tight:
        ui.warn(f"Only ~{candidate.spare_gib:.0f} GiB is left for the KV cache on this GPU")
        ui.hint("If vLLM runs out of memory: lower --max-model-len or add --kv-cache-dtype fp8.")
    guard = guard_text(state)
    if guard:
        ui.step(f"Guard: {guard}")
    else:
        ui.warn("No guard: the pod bills until `ronin down` (max_hours / idle_minutes are 0)")
    print()

    url = proxy_url(pod["id"], cfg.port)
    logs = LogStream(rp, pod["id"], tail=5000, reconnect=True)
    recent: deque[str] = deque(maxlen=300)
    starts, fatal_at = 0, None
    running, last_poll = False, 0.0
    snapshot, last_snapshot = None, 0.0
    progress = ui.Progress("Waiting for RunPod to assign the machine")

    def no_gpu(line: str) -> None:
        if any(m in line for m in NO_GPU_MARKERS):
            progress.stop()
            abort_up(rp, pod["id"], recent, keep=args.keep_on_failure, prefer_reason=True,
                     reason=f"the host's GPU is not usable ({line.strip()[-80:]}). "
                            "A broken host, not your profile: `ronin up` again gets another one")

    try:
        # Short timeouts: on Windows Ctrl+C waits for a blocking socket read to finish.
        while not (running and healthy(url, timeout=3)):
            now = time.monotonic()
            if not running and now - last_poll >= 5:
                last_poll = now
                try:
                    status = rp.get_pod(pod["id"], timeout=10)["status"]
                except RunPodError as e:
                    if not e.transient:
                        raise
                    status = None  # network blip: keep waiting, check again later
                if status == "RUNNING":
                    running = True
                elif status in DEAD_STATES:
                    progress.stop()
                    abort_up(rp, pod["id"], recent, keep=args.keep_on_failure,
                             reason=f"the pod ended in state {status}")
                elif progress.elapsed > POD_READY_TIMEOUT:
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
            # The long-lived stream can bring the container's lines minutes late; a fresh
            # snapshot every SNAPSHOT_EVERY s catches a dead GPU or a fatal error early.
            # Only those markers are read from it: it repeats lines the stream already gave.
            if snapshot and snapshot.done:
                for line in snapshot.lines:
                    if line not in recent:
                        recent.append(line)
                    if fatal_at is None and any(m in line for m in FATAL_MARKERS):
                        fatal_at = time.monotonic()
                    no_gpu(line)
                snapshot = None
            if running and snapshot is None and now - last_snapshot >= SNAPSHOT_EVERY:
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
    finally:
        progress.stop()

    ui.ok(f"Model server ready in {ui.duration(progress.elapsed)}")
    print()
    mascot.show("ready", [
        "Ready. The katana is sharp.",
        f"Model: {profile.model}",
        f"Run `ronin {cfg.client}` to begin.",
    ])


STOCK_POLL = 30                  # seconds between stock checks while waiting for a GPU


# RunPod's wording when the GPU is out of stock (vs. a request it will never accept).
NO_STOCK = ("no longer any instances available", "no instances available",
            "out of stock", "insufficient capacity")


def is_no_stock(e: RunPodError) -> bool:
    return any(s in (e.detail or "").lower() for s in NO_STOCK)


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


def create_with_stock(rp: RunPod, cfg: Config, candidates, clouds: list[str], make_body,
                      max_wait: float = 0, sleep=time.sleep, on_wait=None):
    """Create the pod on the first cloud × GPU with stock, in order. When every attempt
    fails for lack of stock, wait and try again -- nothing is billed while waiting --
    until `max_wait` seconds have passed (0 = until Ctrl+C). Returns (pod, candidate, cloud).

    Errors that are not about stock (an invalid request) are shown once and that GPU is
    still retried with the rest, unless no attempt failed for stock: then waiting is pointless."""
    where = cfg.datacenter or "any datacenter"
    started, shown, progress = time.monotonic(), set(), None
    try:
        while True:
            errors = []
            for cloud in clouds:
                for candidate in candidates:
                    try:
                        pod = rp.create_pod(make_body(candidate.gpu, cloud))
                        return pod, candidate, cloud
                    except RunPodError as e:
                        if e.status in (401, 403):
                            raise
                        errors.append(e)
                        # RunPod's own reason: "no stock" and "invalid request" look different
                        key = (cloud, candidate.gpu, e.detail)
                        if key not in shown and not is_no_stock(e):
                            shown.add(key)
                            if progress:
                                progress.stop()
                            ui.warn(f"{candidate.gpu} ({cloud.lower()}) in {where}: {e.detail}")
            # "no stock" may come back as a 5xx: it is not RunPod being down
            if all(e.transient and not is_no_stock(e) for e in errors):
                raise ConfigError("Could not create the pod: RunPod is not responding\n"
                                  "Try again in a moment.")
            if not any(is_no_stock(e) for e in errors):
                raise ConfigError("Could not create a pod with any GPU in the profile\n"
                                  "RunPod refused the request (see above); waiting would not help.")
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
                progress = ui.Progress(f"Waiting for stock (checking every {STOCK_POLL} s)")
            if on_wait:
                on_wait()
            for _ in range(STOCK_POLL):
                progress.tick()
                sleep(1)
    finally:
        if progress:
            progress.stop()


def abort_up(rp: RunPod, pod_id: str, recent, keep: bool, reason: str | None = None,
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
    rp = client()
    try:
        session = session_cost(rp.get_pod(state["pod_id"]), since=state.get("created_at"))
    except RunPodError:
        session = None
    before = dict(state)            # terminate() clears the pod keys, balance included
    pod_id = terminate(rp, state)
    real = real_session_cost(rp, before)
    detail = (f"The weights remain on the network volume (≈ {monthly(cfg.volume_size_gb)})."
              if cfg.storage == "volume" else "The downloaded weights go with it.")
    lines = [f"Pod {pod_id} deleted.", detail]
    if session:
        lines.append(ui.dim(f"Session: {session}"))
    if real is not None:
        # RunPod debits the last few minutes after the pod is gone (seen: half the total),
        # so right after a delete the balance can trail the estimate
        lines.append(f"Real cost so far: ${real:.2f} {ui.dim('(account balance since `ronin up`)')}")
        estimate = estimated_cost(before)
        if estimate is not None and real < estimate * 0.9:
            lines.append(ui.dim(f"RunPod debits the last minutes later: expect ≈ ${estimate:.2f} "
                                "(`ronin cost` has the final figure)"))
    lines.append("The ronin sheathes the katana.")
    mascot.show("down", lines)


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
    info = {"pod": None, "ready": False, "storage": cfg.storage,
            "volume_id": state.get("volume_id") if cfg.storage == "volume" else None,
            "account": None}
    waiting = waiting_for_stock(state)
    info["waiting"] = ({"profile": waiting["profile"], "clouds": waiting["clouds"],
                        "since": waiting["since"]} if waiting else None)
    try:
        rp = client()
        info["account"] = rp.account()      # nice to have: never a reason for status to fail
    except (ConfigError, RunPodError):
        rp = None
    if not state.get("pod_id"):
        return info
    rp = rp or client()
    try:
        pod = rp.get_pod(state["pod_id"])
    except RunPodError as e:
        if e.status != 404:
            raise
        info["pod"] = {"id": state["pod_id"], "status": "GONE"}
        return info
    uptime = (pod.get("runtime") or {}).get("uptime")
    cost = pod.get("cost")
    # billed time runs from creation (image pull and download included), not container uptime
    billed = time.time() - state["created_at"] if state.get("created_at") else uptime
    url = proxy_url(pod["id"], cfg.port)
    info["ready"] = pod["status"] == "RUNNING" and healthy(url, timeout=5)
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
    if not pod:
        waiting = info.get("waiting")
        lines = [ui.field("Pod", ui.yellow("waiting for stock") if waiting else ui.dim("none")),
                 ui.field("Storage", storage)]
        if account_line:
            lines.append(account_line)
        if waiting:
            mascot.show("wait", [*lines, "", waiting_text(waiting),
                                 "Ctrl+C it in its terminal to cancel."], top=True)
        else:
            mascot.show("rest", [*lines, "", "The ronin rests. `ronin up` to wake it."], top=True)
        return
    if pod["status"] == "GONE":
        mascot.show("rest", [
            ui.field("Pod", f"{pod['id']} {ui.yellow('no longer exists on RunPod')}"),
            "",
            "`ronin down` to clear the local state.",
        ], top=True)
        return
    ok = info["ready"]
    status_color = ui.green if pod["status"] == "RUNNING" else ui.yellow
    cost = "?" if pod["cost_per_hour"] is None else pod["cost_per_hour"]
    lines = [
        ui.field("Pod", f"{pod['id']} {status_color(pod['status'])}"),
        ui.field("Profile", f"{pod['profile']} {ui.dim('· ' + str(pod['model']))}"),
        ui.field("GPU", f"{pod['gpu'] or '?'} {ui.dim('·')} ${cost}/h"),
    ]
    uptime = pod["uptime_seconds"]
    session = session_cost({"runtime": {"uptime": uptime}, "cost": pod["cost_per_hour"]},
                           since=pod.get("created_at"))
    if session:
        if pod.get("session_cost_real") is not None:
            session += f" {ui.dim('·')} real so far ${pod['session_cost_real']:.2f}"
        lines.append(ui.field("Session", session))
    guard = guard_text(state)
    lines.append(ui.field("Guard", guard if guard else ui.yellow("none · bills until `ronin down`")))
    lines += [
        ui.field("URL", ui.dim(pod["url"])),
        ui.field("Server", ui.green("ready") if ok else ui.yellow("not responding yet")),
        ui.field("Storage", storage),
    ]
    if account_line:
        lines.append(account_line)
    if uptime and uptime > WARN_AFTER_HOURS * 3600:
        lines += ["", ui.yellow(f"! Running for {ui.duration(uptime)} and still billing — "
                                "`ronin down` if you are done.")]
    mascot.show("ready" if ok else "wait", lines, top=True)


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
    """What RunPod actually charged, per pod, from its billing API."""
    from datetime import datetime, timedelta, timezone
    rp = client()
    state = load_state()
    since = (datetime.now(timezone.utc) - timedelta(days=args.days - 1)).strftime("%Y-%m-%dT00:00:00Z")
    pods = billed_by_pod(rp.pod_billing(since, bucket="day"))
    history = {h["pod_id"]: h for h in state.get("history", [])}
    print(f"  {ui.bold(f'Billed by RunPod since {since[:10]}')} {ui.dim('(UTC · real amounts)')}\n")
    if pods:
        print("  " + ui.dim(f"{'pod':<16}  {'profile':<16}  {'day':<10}  {'billed':>8}  {'amount':>8}"))
        for pod_id, p in sorted(pods.items(), key=lambda kv: kv[1]["first"]):
            # pad before colouring: ANSI codes would count towards the width
            name = history.get(pod_id, {}).get("profile")
            profile = name.ljust(16) if name else ui.dim("-".ljust(16))
            print(f"  {pod_id:<16}  {profile}  {p['first'][:10]:<10}  "
                  f"{ui.duration(p['minutes'] * 60):>8}  {'$' + format(p['amount'], '.2f'):>8}")
        print(f"\n  Total billed: {ui.bold(f'${sum(p['amount'] for p in pods.values()):.2f}')}")
    else:
        ui.step("Nothing billed in this period yet.")
    # pods ronin created that billing does not show yet (it is published ~1 h late)
    cutoff = datetime.fromisoformat(since.replace("Z", "+00:00")).timestamp()
    pending = [h for h in history.values()
               if h.get("created_at", 0) >= cutoff and h["pod_id"] not in pods]
    if pending:
        print()
        ui.warn(f"{len(pending)} pod(s) not in RunPod's billing yet (it is published about an hour late):")
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
    except RunPodError as e:
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
        logs = LogStream(client(), state["pod_id"], tail=args.tail, reconnect=False)
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
    logs = LogStream(client(), state["pod_id"], tail=args.tail)
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
                                  secure=args.secure, max_wait=args.max_wait), cfg)
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


def _pod_registered() -> bool:
    try:
        return bool(load_state().get("pod_id"))
    except ConfigError:
        return False


GLOBAL_VALUED = ("--config",)               # global options that take a value


LAUNCH_COMMANDS = ("opencode", "claude")    # verbs that forward trailing args to the agent


AGENT_FLAGS = ("--up", "--secure", "-h", "--help")   # ronin's own launch options without a value


AGENT_VALUED = ("--profile", "--max-wait")           # ... and with one


def add_stock_options(p) -> None:
    p.add_argument("--secure", action="store_true",
                   help="if the configured cloud (community) has no stock, use secure cloud "
                        "(pricier) instead of waiting")
    p.add_argument("--max-wait", type=float, metavar="MINUTES",
                   help="give up after waiting this long for stock (default: wait until Ctrl+C)")


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

    p = sub.add_parser("volume", help="list, create or delete network volumes")
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
    p.set_defaults(func=cmd_gpus)

    p = sub.add_parser("up", help="bring the pod up with vLLM and wait until it is ready")
    p.add_argument("-p", "--profile", help="profile from ronin.toml (default: the one in [defaults])")
    p.add_argument("--keep-on-failure", action="store_true",
                   help="if vLLM fails to start, do not delete the pod (to investigate with `ronin logs`)")
    p.add_argument("--max-hours", type=float,
                   help="the pod deletes itself after this many hours (default: max_hours in ronin.toml; 0 = never)")
    p.add_argument("--idle", type=float, metavar="MINUTES",
                   help="the pod deletes itself after this many minutes without requests "
                        "(default: idle_minutes in ronin.toml; 0 = never)")
    p.add_argument("--skip-memory-check", action="store_true",
                   help="try the profile's GPUs even if the weights look too big for them")
    add_stock_options(p)
    p.set_defaults(func=cmd_up)

    sub.add_parser("down", help="delete the pod (the volume is kept)").set_defaults(func=cmd_down)
    p = sub.add_parser("cost", help="what RunPod actually billed, per pod, and the account balance")
    p.add_argument("--days", type=int, default=7, help="days back, today included (default 7)")
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
                   help=f"seconds between RunPod checks (default {WATCHDOG_POLL})")
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
        code = args.func(args, None if args.command in OWN_CONFIG else load_config()) or 0
    except (ConfigError, RunPodError) as e:
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
    if padded:
        sys.stderr.flush()
        print()
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
