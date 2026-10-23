"""Pod plumbing shared by the commands: RunPod requests, the vLLM endpoint, the pod
definition, log streaming and the client config that points the coding agent at the pod."""

from __future__ import annotations

import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import USER_AGENT, config, ui
from .config import Config, ConfigError, load_state, require_env, save_state
from .runpod import RunPod, RunPodError

VOLUME_MOUNT = "/models"


EPHEMERAL_CACHE = "/root/.cache/huggingface"


GUARD_SOURCE = Path(__file__).with_name("guard.py")


VOLUME_PRICE_GB_MONTH = 0.07     # RunPod network volume, $/GB/month (approximate)


# Pod states where the container is no longer serving (it may still hold its disk).
DEAD_STATES = ("TERMINATED", "EXITED", "ERROR")


# State keys that describe the registered pod.
POD_KEYS = ("pod_id", "api_key", "profile", "model", "deadline", "idle_minutes",
            "created_at", "balance_at_up")


# Pods ronin created, newest last (state["history"]), so `ronin cost` can name the
# profile behind each billed pod id. Kept across pods; capped.
HISTORY_MAX = 100


# Log lines that mean vLLM will not start.
FATAL_MARKERS = ("EngineCore failed to start", "Engine core initialization failed")


# Log lines that mean the host's GPU is unusable (seen on community hosts): llama.cpp
# would carry on in CPU and look healthy while crawling, so ronin stops at once.
NO_GPU_MARKERS = ("failed to initialize CUDA", "no usable GPU found",
                  "No CUDA GPUs are available", "CUDA driver initialization failed")


# RunPod restarts the container when the process dies; seeing it twice = crash loop.
CONTAINER_START = "start container for"


ERROR_LINE = re.compile(r"\b\w+(Error|Exception): ")


# Exceptions that only wrap the real cause; used only when there is nothing better.
GENERIC_ERRORS = ("Engine core initialization failed",)


def proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def healthy(url: str, timeout: float = 10) -> bool:
    req = urllib.request.Request(url + "/health", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def client() -> RunPod:
    return RunPod(require_env("RUNPOD_API_KEY"))


class LogStream:
    """Read the pod logs on a daemon thread and hand them off through a queue.

    On Windows a blocking socket read is not interrupted by Ctrl+C, so the stream is
    never read from the main thread. With reconnect=True, a dropped stream is reopened
    without backfill (so lines are not duplicated).
    """

    DONE = object()

    def __init__(self, rp: RunPod, pod_id: str, tail: int, reconnect: bool = False):
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, args=(rp, pod_id, tail, reconnect), daemon=True).start()

    def _run(self, rp: RunPod, pod_id: str, tail: int, reconnect: bool) -> None:
        while True:
            try:
                for event in rp.stream_logs(pod_id, tail=tail):
                    self.lines.put(event.get("line", ""))
            except Exception as e:
                if not reconnect:
                    self.lines.put(e)
                    break
            if not reconnect:
                break
            tail = 0
            time.sleep(5)
        self.lines.put(self.DONE)

    def drain(self) -> list[str]:
        """Lines available right now, without blocking."""
        out = []
        while True:
            try:
                item = self.lines.get_nowait()
            except queue.Empty:
                return out
            if isinstance(item, str):
                out.append(item)


class LogSnapshot:
    """The pod's latest log lines, read off a fresh stream on a daemon thread.

    A long-lived stream can deliver the container's lines minutes late, while a new one
    backfills them at once; the stream never ends by itself, so it is closed after
    `wait` seconds. Poll `done` / `lines` from the main thread (never blocks)."""

    def __init__(self, rp: RunPod, pod_id: str, tail: int = 200, wait: float = 4.0):
        self.lines: list[str] = []
        self.done = False
        threading.Thread(target=self._run, args=(rp, pod_id, tail, wait), daemon=True).start()

    def _run(self, rp: RunPod, pod_id: str, tail: int, wait: float) -> None:
        resp = None
        try:
            resp = rp.open_logs(pod_id, tail)
            timer = threading.Timer(wait, resp.close)
            timer.daemon = True
            timer.start()
            for event in rp.stream_logs(pod_id, resp=resp):
                self.lines.append(event.get("line", ""))
        except Exception:
            pass            # a snapshot is best effort; the main stream still runs
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            self.done = True


def terminate(rp: RunPod, state: dict) -> str | None:
    """Delete the registered pod and clear it from state. Returns its id."""
    pod_id = state.get("pod_id")
    if not pod_id:
        return None
    try:
        rp.delete_pod(pod_id)
    except RunPodError as e:
        if e.status != 404:
            raise
    for entry in state.get("history", []):
        if entry.get("pod_id") == pod_id and not entry.get("deleted_at"):
            entry["deleted_at"] = time.time()
    clear_pod(state)
    return pod_id


def terminate_own(rp: RunPod, pod_id: str) -> None:
    """Delete `pod_id` -- the pod this long-running process (`up`, `watchdog`) created or
    guards -- re-reading the state first. Its in-memory copy may be old: if `ronin down`
    and a new `up` ran meanwhile, saving it would unregister the new pod, which would then
    bill without `ronin status` seeing it. The state changes only where it is about `pod_id`."""
    state = load_state()
    if state.get("pod_id") == pod_id:
        terminate(rp, state)
        return
    try:
        rp.delete_pod(pod_id)
    except RunPodError as e:
        if e.status != 404:
            raise
    marked = False
    for entry in state.get("history", []):
        if entry.get("pod_id") == pod_id and not entry.get("deleted_at"):
            entry["deleted_at"] = time.time()
            marked = True
    if marked:
        save_state(state)


def forget_own(pod_id: str) -> None:
    """Clear `pod_id` from the state if it is still the registered pod (see terminate_own)."""
    state = load_state()
    if state.get("pod_id") == pod_id:
        clear_pod(state)


def remember_pod(state: dict, pod: dict, profile_name: str) -> None:
    """Add a just-created pod to the history (saved with the next save_state)."""
    history = state.setdefault("history", [])
    history.append({"pod_id": pod["id"], "profile": profile_name,
                    "gpu": (pod.get("gpu") or {}).get("id"),
                    "cost_per_hour": pod.get("cost"), "created_at": time.time()})
    del history[:-HISTORY_MAX]


def clear_pod(state: dict) -> None:
    """Forget the registered pod (the volume id is kept)."""
    for key in POD_KEYS:
        state.pop(key, None)
    save_state(state)


def root_cause(lines) -> str | None:
    """Last exception in the log (e.g. 'ValueError: ...'), preferring the real cause."""
    errors = []
    for line in lines:
        match = ERROR_LINE.search(line)
        if match:
            errors.append(line[match.start():].strip())
    specific = [e for e in errors if not any(g in e for g in GENERIC_ERRORS)]
    return (specific or errors or [None])[-1]


def guard_text(state: dict) -> str | None:
    """'deletes itself at 02:30 (in 7h 58m) · or after 60m idle', or None if unguarded."""
    parts = []
    deadline = state.get("deadline") or 0
    if deadline:
        left = max(0, deadline - time.time())
        parts.append(f"deletes itself at {time.strftime('%H:%M', time.localtime(deadline))} "
                     f"(in {ui.duration(left)})")
    idle = state.get("idle_minutes") or 0
    if idle:
        parts.append(f"after {ui.duration(idle * 60)} idle")
    return " · or ".join(parts) or None


def session_cost(pod: dict, since: float | None = None) -> str | None:
    """'1h 12m · ≈ $1.31 (estimate)' from the hourly cost and the time billed so far.

    RunPod bills from the pod's creation, so `since` (the creation time ronin recorded)
    is used when known; the container's uptime leaves out the image pull and the model
    download, and under-counts. Either way it is an estimate: see `ronin cost`."""
    seconds = time.time() - since if since else (pod.get("runtime") or {}).get("uptime")
    cost = pod.get("cost")
    if seconds is None:
        return None
    text = ui.duration(seconds)
    if isinstance(cost, (int, float)):
        text += f" · ≈ ${cost * seconds / 3600:.2f} (estimate)"
    return text


def real_session_cost(rp: RunPod, state: dict) -> float | None:
    """Money the account lost since `ronin up` (balance then minus balance now): the
    real figure, available at once -- billing per pod is only published ~1 h later.
    It includes anything else billing on the account meanwhile (other pods, volumes)."""
    before = state.get("balance_at_up")
    if before is None:
        return None
    try:
        return max(0.0, before - rp.account()["balance"])
    except RunPodError:
        return None


def ensure_volume(rp: RunPod, cfg: Config, state: dict) -> str:
    """Return the network volume id, creating it if needed."""
    if not cfg.datacenter:
        raise ConfigError("Missing datacenter for the volume\n"
                          "Set it in ronin.toml (`ronin gpus` shows where there is stock).")
    volumes = rp.list_volumes()
    for vol in volumes:
        if vol["name"] == cfg.volume_name and vol["dataCenter"] == cfg.datacenter:
            state["volume_id"] = vol["id"]
            save_state(state)
            return vol["id"]
    ui.step(f"Creating network volume {cfg.volume_name} ({cfg.volume_size_gb} GB) in {cfg.datacenter} "
            f"{ui.dim('· ≈ ' + monthly(cfg.volume_size_gb))}")
    vol = rp.create_volume(cfg.volume_name, cfg.volume_size_gb, cfg.datacenter)
    state["volume_id"] = vol["id"]
    save_state(state)
    # A changed datacenter leaves the old volume behind, still billing every month.
    for old in volumes:
        if old["name"] == cfg.volume_name:
            ui.warn(f"Volume {old['id']} ({old['dataCenter']}) has the same name and still bills "
                    f"≈ {monthly(old.get('size', 0))}")
            ui.hint(f"`ronin volume rm {old['id']}` if you no longer need it")
    return vol["id"]


def monthly(size_gb: float) -> str:
    return f"${size_gb * VOLUME_PRICE_GB_MONTH:.2f}/month"


# Default container image per backend. vLLM's is pinned in ronin.toml ([defaults].image);
# llama.cpp's floats on the server-cuda tag -- pin it per profile for reproducibility.
LLAMACPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"


# Lowest host CUDA version per backend, unless the profile sets min_cuda. The llama.cpp
# image declares NVIDIA_REQUIRE_CUDA=cuda>=12.8; vLLM's is left unfiltered.
BACKEND_MIN_CUDA = {"llamacpp": "12.8"}


# Prometheus metric names the guard watches, when the backend is not vLLM. vLLM's names
# are guard.py's own defaults, so only the others need to be passed through the env.
GUARD_METRICS = {
    "llamacpp": {
        "RONIN_BUSY_METRICS": "llamacpp:requests_processing,llamacpp:requests_deferred",
        "RONIN_DONE_METRIC": "llamacpp:n_decode_total",
    },
}


def _vllm_spec(cfg: Config, profile, volume_id: str | None, api_key: str):
    """(env, entrypoint, cmd, image) to serve `profile` with vLLM's OpenAI server."""
    env = {
        "HF_HOME": f"{VOLUME_MOUNT}/hf" if volume_id else EPHEMERAL_CACHE,
        # parallel downloads with hf_xet: matters mostly in ephemeral mode
        "HF_XET_HIGH_PERFORMANCE": "1",
        # an env var rather than --api-key keeps the key out of the container command line
        "VLLM_API_KEY": api_key,
    }
    if volume_id:
        # Persist vLLM's torch.compile cache on the volume so later boots skip the
        # ~80 s recompile. In ephemeral mode it lives on the container disk and is
        # wiped on `ronin down`, so there is nothing to persist.
        env["VLLM_CACHE_ROOT"] = f"{VOLUME_MOUNT}/vllm"
    cmd = [
        profile.model,
        "--host", "0.0.0.0",
        "--port", str(cfg.port),
        "--served-model-name", cfg.served_name,
        *profile.vllm_args,
    ]
    # Spread one model over several GPUs, unless the profile sets the flag itself.
    if profile.gpu_count > 1 and not serve_arg(profile, "--tensor-parallel-size", "-tp"):
        cmd += ["--tensor-parallel-size", str(profile.gpu_count)]
    if profile.chat_template:
        # vLLM accepts the template as a string (it contains braces): no files to upload
        path = config.project.root / profile.chat_template
        if not path.exists():
            raise ConfigError(f"Chat template not found\n{path}")
        cmd += ["--chat-template", path.read_text(encoding="utf-8")]
    return env, ["vllm", "serve"], cmd, profile.image or cfg.image


def _llamacpp_spec(cfg: Config, profile, volume_id: str | None, api_key: str):
    """(env, entrypoint, cmd, image) to serve a GGUF `profile` with llama-server.

    llama-server exposes the same OpenAI `/v1` and `/health` as vLLM, so nothing
    downstream (the coding agent, the guard's health check) changes. `--jinja` turns on
    OpenAI tool-calling; `--metrics` exposes `/metrics` for the idle guard."""
    env = {
        "HF_HOME": f"{VOLUME_MOUNT}/hf" if volume_id else EPHEMERAL_CACHE,
        "HF_XET_HIGH_PERFORMANCE": "1",
        # llama-server reads the API key from LLAMA_API_KEY: keeps it off the command line
        "LLAMA_API_KEY": api_key,
    }
    if volume_id:
        # Persist the downloaded GGUFs on the volume (ephemeral keeps them on the disk).
        env["LLAMA_CACHE"] = f"{VOLUME_MOUNT}/llama"
    cmd = ["-hf", profile.model]
    if profile.hf_file:
        cmd += ["-hff", profile.hf_file]
    cmd += [
        "--host", "0.0.0.0",
        "--port", str(cfg.port),
        "--alias", cfg.served_name,   # the model name llama-server reports (what opencode uses)
        "--metrics",                  # expose /metrics so the idle guard can see activity
        "--jinja",                    # OpenAI-style tool-calling from the model's chat template
        *profile.llamacpp_args,
    ]
    # The image's PATH does not include /app, where the binary lives.
    return env, [LLAMACPP_BIN], cmd, profile.image or LLAMACPP_IMAGE


LLAMACPP_BIN = "/app/llama-server"


# The llama.cpp image has no Python, which the guard needs: install it on boot (a few
# seconds, next to minutes of model download), then run the guard, whose source arrives
# in RONIN_GUARD_SRC to dodge shell quoting. If Python cannot be installed the container
# exits, so `ronin up` sees the failure and deletes the pod -- never an unguarded pod.
LLAMACPP_GUARD_SH = (
    'command -v python3 >/dev/null 2>&1 || { echo "[ronin-guard] installing python3"; '
    'apt-get update -qq && apt-get install -y -qq --no-install-recommends '
    'python3 ca-certificates >/dev/null; } || exit 1; '
    'exec python3 -c "$RONIN_GUARD_SRC" "$@"'
)


def _guard_wrap(backend: str, entrypoint: list[str], env: dict) -> list[str]:
    """The entrypoint that runs the server under the in-pod guard (see guard.py)."""
    source = GUARD_SOURCE.read_text(encoding="utf-8")
    if backend == "llamacpp":
        env["RONIN_GUARD_SRC"] = source
        return ["sh", "-c", LLAMACPP_GUARD_SH, "ronin-guard", *entrypoint]
    return ["python3", "-c", source, *entrypoint]


BACKENDS = {"vllm": _vllm_spec, "llamacpp": _llamacpp_spec}


def pod_body(cfg: Config, profile, gpu_id: str, volume_id: str | None, api_key: str,
             deadline: float = 0, idle_seconds: float = 0, cloud: str | None = None) -> dict:
    backend = getattr(profile, "backend", "vllm")
    env, entrypoint, cmd, image = BACKENDS[backend](cfg, profile, volume_id, api_key)
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if deadline or idle_seconds:
        # the guard wraps the server and deletes the pod from inside (see guard.py)
        env.update(RONIN_DEADLINE=str(int(deadline)), RONIN_IDLE=str(int(idle_seconds)),
                   RONIN_PORT=str(cfg.port))
        env.update(GUARD_METRICS.get(backend, {}))
        entrypoint = _guard_wrap(backend, entrypoint, env)
    gpu = {"id": gpu_id, "count": profile.gpu_count}
    min_cuda = getattr(profile, "min_cuda", None) or BACKEND_MIN_CUDA.get(backend)
    if min_cuda:
        # only hosts whose driver supports this CUDA version (RunPod filters on create)
        gpu["minCudaVersion"] = min_cuda
    body = {
        "name": f"ronin-{profile.name}",
        "image": image,
        "cloud": cloud or cfg.cloud,
        "gpu": gpu,
        # with a volume the pod must go to its datacenter; in ephemeral only if one is set
        "dataCenterIds": [cfg.datacenter] if cfg.datacenter else [],
        "disk": 30 if volume_id else profile.disk_gb,
        "ports": [f"{cfg.port}/http"],
        "env": env,
        "entrypoint": entrypoint,
        "cmd": cmd,
    }
    if volume_id:
        body["mounts"] = {"network": [{"volumeId": volume_id, "path": VOLUME_MOUNT}]}
    return body


CLAUDE_MAX_OUTPUT = 16384       # default 32000 for unrecognised models: eats too much context


DEFAULT_CONTEXT = 131072


def serve_args(profile) -> list[str]:
    """The profile's serving arguments for its backend (vllm_args or llamacpp_args)."""
    if not profile:
        return []
    if getattr(profile, "backend", "vllm") == "llamacpp":
        return profile.llamacpp_args
    return profile.vllm_args


def serve_arg(profile, *names: str) -> str | None:
    """Value of a serving option in the profile (`--name value` or `--name=value`)."""
    args = serve_args(profile)
    for i, arg in enumerate(args):
        if arg in names and i + 1 < len(args):
            return args[i + 1]
        name, eq, value = arg.partition("=")
        if eq and name in names:
            return value
    return None


# Backwards-compatible alias: historically this only read vLLM args.
vllm_arg = serve_arg


def max_model_len(profile) -> int:
    """The profile's context length: --max-model-len (vLLM) or -c/--ctx-size (llama.cpp)."""
    if profile and getattr(profile, "backend", "vllm") == "llamacpp":
        value = serve_arg(profile, "-c", "--ctx-size")
    else:
        value = serve_arg(profile, "--max-model-len")
    return parse_tokens(value) if value else DEFAULT_CONTEXT


def parse_tokens(text: str) -> int:
    """vLLM accepts '131072' as well as '128K' / '1M' (binary) and '100k' (decimal)."""
    units = {"k": 1000, "m": 1000 ** 2, "K": 1024, "M": 1024 ** 2}
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


# The key is referenced from opencode.json as {env:RONIN_API_KEY} and passed through the
# environment at launch, so the secret is never written to the config file.
OPENCODE_API_KEY_ENV = "RONIN_API_KEY"


def opencode_config(cfg: Config, state: dict) -> dict:
    """The opencode.json content that points opencode at the pod's OpenAI endpoint.

    opencode talks OpenAI-compatible `/v1/chat/completions` (via @ai-sdk/openai-compatible),
    which vLLM already serves on the same port as its Anthropic one -- so no proxy is needed
    and the pod is unchanged. A custom provider needs its context limit and tool support
    declared explicitly: opencode cannot autodetect them."""
    if not state.get("pod_id") or not state.get("api_key"):
        raise ConfigError("No pod is active\nRun `ronin up` first (or `ronin opencode --up`).")
    url = proxy_url(state["pod_id"], cfg.port)
    context = max_model_len(cfg.profiles.get(state.get("profile", "")))
    name = cfg.served_name
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"ronin/{name}",
        "provider": {
            "ronin": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "ronin (self-hosted)",
                "options": {
                    "baseURL": f"{url}/v1",
                    "apiKey": f"{{env:{OPENCODE_API_KEY_ENV}}}",
                },
                "models": {
                    name: {
                        "name": name,
                        # agentic coding needs tool use; vLLM exposes it via
                        # --enable-auto-tool-choice --tool-call-parser in the profile
                        "tools": True,
                        "limit": {"context": context, "output": CLAUDE_MAX_OUTPUT},
                    },
                },
            },
        },
    }


def claude_env(cfg: Config, state: dict) -> dict[str, str]:
    if not state.get("pod_id") or not state.get("api_key"):
        raise ConfigError("No pod is active\nRun `ronin up` first (or `ronin claude --up`).")
    url = proxy_url(state["pod_id"], cfg.port)
    context = max_model_len(cfg.profiles.get(state.get("profile", "")))
    # Claude Code does not know "ronin" and would assume 200K of context: tell it the
    # real value and compact with headroom for output (the minimum allowed is 100000).
    compact_at = max(100_000, context - CLAUDE_MAX_OUTPUT)
    return {
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context),
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(compact_at),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(CLAUDE_MAX_OUTPUT),
        # a non-Anthropic backend receives this block as prompt text
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        # vLLM does not implement the Anthropic API beta fields
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "ANTHROPIC_BASE_URL": url,
        # vLLM checks `Authorization: Bearer`, which is what ANTHROPIC_AUTH_TOKEN sends.
        # ANTHROPIC_API_KEY is left unset (and removed from the inherited env): with both
        # set at once Claude Code warns about ambiguous authentication.
        "ANTHROPIC_AUTH_TOKEN": state["api_key"],
        "ANTHROPIC_MODEL": cfg.served_name,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": cfg.served_name,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": cfg.served_name,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": cfg.served_name,
        "CLAUDE_CODE_SUBAGENT_MODEL": cfg.served_name,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def pod_problem(cfg: Config, state: dict) -> str | None:
    """Why the registered pod cannot serve Claude Code, or None if vLLM answers.
    A pod that no longer exists (or is dead and then deleted) is reported as 'gone…'."""
    if healthy(proxy_url(state["pod_id"], cfg.port), timeout=5):
        return None
    rp = client()
    try:
        pod = rp.get_pod(state["pod_id"])
    except RunPodError as e:
        if e.status != 404:
            raise
        clear_pod(state)
        return "gone (it no longer exists on RunPod; the guard may have deleted it)"
    if pod["status"] in DEAD_STATES:
        terminate(rp, state)
        return f"gone (it was {pod['status']}; deleted)"
    return f"{pod['status']} but vLLM is not responding yet"
