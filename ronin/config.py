"""Configuration (ronin.toml), secrets (.env) and local state (.ronin/state.json).

The project is the directory holding ronin.toml. It is found from, in order: the
--config flag, the RONIN_CONFIG variable, the current directory and its parents, and
finally the source checkout ronin runs from. `.env` and `.ronin/` live next to it, so
each project has its own secrets and its own pod.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "ronin.toml"
# The checkout ronin is installed from (editable install); a fallback, not a default.
SOURCE_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    pass


# Written by `ronin init` when the project has no ronin.toml yet.
STARTER_CONFIG = """\
# ronin configuration. Secrets (RUNPOD_API_KEY, VAST_API_KEY, HF_TOKEN) live in .env, not here.

[defaults]
profile = "main"
# Where pods run: "runpod" or "vast" (vast.ai). `ronin up --provider vast` overrides it.
provider = "runpod"
image = "vllm/vllm-openai:v0.32.0"
# "ephemeral": weights downloaded on every `ronin up` (no fixed cost).
# "volume": persistent network volume (fast boots, ~$0.07/GB/month, pinned datacenter).
# RunPod only: vast.ai has no network volumes.
storage = "ephemeral"
# RunPod datacenter. Empty = any datacenter with stock (required with a volume). See `ronin gpus`.
datacenter = ""
volume_name = "ronin-models"
volume_size_gb = 120
# SECURE: RunPod's own datacenters / vast.ai's datacenter hosts. COMMUNITY: cheaper hosts.
cloud = "SECURE"
port = 8000
served_name = "ronin"
# Coding agent `ronin opencode` launches: "opencode" or "claude".
client = "opencode"
# The pod deletes itself after max_hours of uptime, or after idle_minutes without
# requests, even if your PC is off. 0 disables each one.
max_hours = 8
idle_minutes = 60

# A profile = model + GPUs (tried in order until one has stock) + serving args.
# GPUs are named the same on every provider ("RTX 4090", "L40S"; RunPod ids also work).
# Another profile can reuse it with `extends = "main"` and override any key,
# or append arguments with `extra_vllm_args = [...]`.
# Set `backend = "llamacpp"` for a GGUF model served with llama-server; then put
# its flags in `llamacpp_args` (and the quant in `model`, e.g. "org/repo-GGUF:Q4_K_M").
[profiles.main]
model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
gpus = ["NVIDIA L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition"]
vllm_args = [
  "--max-model-len", "131072",
  "--gpu-memory-utilization", "0.92",
  "--enable-auto-tool-choice",
  "--tool-call-parser", "qwen3_coder",
  "--enable-prefix-caching",
]
"""

STARTER_ENV = """\
# RunPod API key: https://console.runpod.io/user/settings
RUNPOD_API_KEY=
# vast.ai API key (provider = "vast"): https://cloud.vast.ai/manage-keys/
VAST_API_KEY=
# Optional: Hugging Face token (gated models, fewer rate limits)
HF_TOKEN=
"""


class Project:
    """Paths of the active project; set once by `use_project()`."""

    def __init__(self, config: Path):
        self.config = config.resolve()
        self.root = self.config.parent
        self.env = self.root / ".env"
        self.state_dir = self.root / ".ronin"
        self.state = self.state_dir / "state.json"


def find_config(explicit: str | None = None, start: Path | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    if os.environ.get("RONIN_CONFIG"):
        return Path(os.environ["RONIN_CONFIG"])
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / CONFIG_NAME).is_file():
            return folder / CONFIG_NAME
    if (SOURCE_ROOT / CONFIG_NAME).is_file():
        return SOURCE_ROOT / CONFIG_NAME
    return None


def use_project(explicit: str | None = None) -> Project:
    """Select the project (see the module docstring). Without a ronin.toml anywhere,
    the current directory is used, so `ronin init` can create one there."""
    global project
    found = find_config(explicit)
    project = Project(found if found else Path.cwd() / CONFIG_NAME)
    return project


project = Project(find_config() or Path.cwd() / CONFIG_NAME)


@dataclass
class Profile:
    name: str
    model: str
    gpus: list[str]
    vllm_args: list[str] = field(default_factory=list)
    # Container disk in GB. In ephemeral mode the whole model must fit here.
    disk_gb: int = 80
    # GPUs per pod. >1 serves one model across them with tensor parallelism
    # (ronin adds --tensor-parallel-size). The cost is multiplied by this.
    gpu_count: int = 1
    # Custom chat template (path relative to ronin.toml); passed inline to vLLM.
    chat_template: str | None = None
    # Serving backend: "vllm" (safetensors/FP8/...) or "llamacpp" (GGUF with llama-server).
    # Both expose OpenAI /v1 and /health, so the coding agent and the guard are unchanged.
    backend: str = "vllm"
    # llama-server arguments, the GGUF counterpart of vllm_args (ignored by the vLLM backend).
    llamacpp_args: list[str] = field(default_factory=list)
    # Exact GGUF filename (llama-server -hff), when the `model` repo:quant is not enough.
    hf_file: str | None = None
    # Container image override. Without it, each backend uses its own default image.
    image: str | None = None
    # Lowest host CUDA version ("12.8"): the pod is only placed on machines with at
    # least this driver. Without it, the backend's default (llama.cpp's image needs 12.8).
    min_cuda: str | None = None


@dataclass
class Config:
    image: str
    storage: str          # "ephemeral" (download on each boot) or "volume" (network volume)
    datacenter: str
    volume_name: str
    volume_size_gb: int
    cloud: str
    port: int
    served_name: str
    # Coding agent to launch ("opencode" or "claude") and its executable name.
    client: str
    opencode_bin: str
    # In-pod guard (0 disables each): the pod deletes itself after this many hours
    # of uptime, or once vLLM has served no request for this many minutes.
    max_hours: float
    idle_minutes: float
    default_profile: str
    profiles: dict[str, Profile]
    # Where pods run: "runpod" or "vast" (vast.ai).
    provider: str = "runpod"

    def profile(self, name: str | None) -> Profile:
        name = name or self.default_profile
        if name not in self.profiles:
            raise ConfigError(f"Unknown profile '{name}'\n"
                              f"Available: {', '.join(self.profiles)} (`ronin profiles`)")
        return self.profiles[name]


def resolve_profile(name: str, raw: dict[str, dict], seen: tuple[str, ...] = ()) -> dict:
    """The profile's table with `extends` applied: keys of the base profile, then its
    own; `extra_vllm_args` is appended to the inherited `vllm_args`."""
    if name in seen:
        raise ConfigError(f"Profiles extend each other in a loop: {' -> '.join((*seen, name))}")
    own = dict(raw[name])
    base_name = own.pop("extends", None)
    if base_name is None:
        merged = own
    else:
        if base_name not in raw:
            raise ConfigError(f"Profile '{name}' extends unknown profile '{base_name}'")
        merged = {**resolve_profile(base_name, raw, (*seen, name)), **own}
    extra = merged.pop("extra_vllm_args", [])
    merged["vllm_args"] = [*merged.get("vllm_args", []), *extra]
    extra_llama = merged.pop("extra_llamacpp_args", [])
    merged["llamacpp_args"] = [*merged.get("llamacpp_args", []), *extra_llama]
    return merged


# What ronin.toml may hold, and the kind of value each key takes. Anything else is
# reported (a typo, or a key from another ronin version) instead of silently ignored.
TEXT, INT, NUMBER, ARGS, NAMES = "text", "an integer", "a number", "a list of arguments", "a list of names"
VERSION = "a version like \"12.8\""

DEFAULT_KEYS = {
    "profile": TEXT, "provider": TEXT, "image": TEXT, "storage": TEXT, "datacenter": TEXT,
    "volume_name": TEXT, "volume_size_gb": INT, "cloud": TEXT, "port": INT,
    "served_name": TEXT, "client": TEXT, "opencode_bin": TEXT,
    "max_hours": NUMBER, "idle_minutes": NUMBER,
}

PROFILE_KEYS = {
    "extends": TEXT, "model": TEXT, "gpus": NAMES, "backend": TEXT,
    "vllm_args": ARGS, "extra_vllm_args": ARGS, "llamacpp_args": ARGS, "extra_llamacpp_args": ARGS,
    "disk_gb": INT, "gpu_count": INT, "chat_template": TEXT, "hf_file": TEXT, "image": TEXT,
    "min_cuda": VERSION,
}

# Allowed values, and numeric ranges (low, high; None = unbounded).
CHOICES = {"provider": ("runpod", "vast"), "storage": ("ephemeral", "volume"),
           "cloud": ("SECURE", "COMMUNITY"), "client": ("opencode", "claude"),
           "backend": ("vllm", "llamacpp")}
RANGES = {"port": (1, 65535), "volume_size_gb": (1, None), "disk_gb": (1, None),
          "gpu_count": (1, 8), "max_hours": (0, None), "idle_minutes": (0, None)}

# Serving args of the other backend: set in a profile's own table, they would be ignored.
OTHER_BACKEND_ARGS = {"vllm": ("llamacpp_args", "extra_llamacpp_args"),
                      "llamacpp": ("vllm_args", "extra_vllm_args", "chat_template")}


def _kind_ok(value, kind: str) -> bool:
    plain = (str, int, float)
    if kind == TEXT:
        return isinstance(value, str)
    if kind == VERSION:
        return isinstance(value, (str, float))       # "12.8" or 12.8
    if kind == INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == ARGS:
        return isinstance(value, list) and all(isinstance(a, plain) and not isinstance(a, bool)
                                               for a in value)
    if kind == NAMES:
        return isinstance(value, list) and all(isinstance(a, str) and a.strip() for a in value)
    return True


def _unknown(key: str, where: str, known: dict, other: dict, other_where: str) -> str:
    msg = f"{where}: unknown key '{key}'"
    if key in other:
        return f"{msg} (it belongs in {other_where})"
    close = difflib.get_close_matches(key, known, n=1)
    return f"{msg} (did you mean '{close[0]}'?)" if close else msg


def _check_table(table: dict, where: str, known: dict, other: dict, other_where: str) -> list[str]:
    problems = []
    for key, value in table.items():
        if key not in known:
            problems.append(_unknown(key, where, known, other, other_where))
            continue
        kind = known[key]
        if not _kind_ok(value, kind):
            if kind == ARGS and isinstance(value, str):
                problems.append(f"{where}: {key} must be a list, e.g. "
                                f"{json.dumps(value.split())} (one string is one argument)")
            else:
                problems.append(f"{where}: {key} must be {kind}, not {json.dumps(value)}")
            continue
        if key in CHOICES and value not in CHOICES[key]:
            options = " or ".join(f"'{c}'" for c in CHOICES[key])
            problems.append(f"{where}: {key} must be {options}, not '{value}'")
        if key in RANGES:
            low, high = RANGES[key]
            if value < low or (high is not None and value > high):
                bound = f"between {low} and {high}" if high is not None else f"at least {low}"
                problems.append(f"{where}: {key} must be {bound}, not {value}")
        if key == "min_cuda" and not re.fullmatch(r"\d+\.\d+", str(value)):
            problems.append(f"{where}: min_cuda must look like \"12.8\" (major.minor), not '{value}'")
        if kind == ARGS:
            for arg in value:
                # "--max-model-len 4096" as one item reaches the server as a single argument
                if isinstance(arg, str) and arg.startswith("-") and " " in arg.strip():
                    problems.append(f"{where}: {key} item \"{arg}\" holds several arguments; "
                                    f"split it: {json.dumps(arg.split())}")
    return problems


def validate(raw: dict) -> list[str]:
    """Every problem in a parsed ronin.toml (empty when it is sound), checked before
    anything is built from it, so a typo is reported up front, not as a failed pod."""
    problems = []
    for key in raw:
        if key not in ("defaults", "profiles"):
            close = difflib.get_close_matches(key, ("defaults", "profiles"), n=1)
            hint = f" (did you mean [{close[0]}]?)" if close else " (only [defaults] and [profiles.*])"
            problems.append(f"unknown table [{key}]{hint}")
    d = raw.get("defaults", {})
    if not isinstance(d, dict):
        return problems + ["defaults must be a table: [defaults]"]
    problems += _check_table(d, "[defaults]", DEFAULT_KEYS, PROFILE_KEYS, "a [profiles.<name>] table")
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict) or not all(isinstance(p, dict) for p in profiles.values()):
        return problems + ["profiles must be tables: [profiles.<name>]"]
    if not profiles:
        problems.append("no profiles: add at least one [profiles.<name>] with model and gpus")
    for name, table in profiles.items():
        where = f"[profiles.{name}]"
        problems += _check_table(table, where, PROFILE_KEYS, DEFAULT_KEYS, "[defaults]")
        base = table.get("extends")
        if isinstance(base, str) and base not in profiles:
            problems.append(f"{where}: extends unknown profile '{base}'")
    if problems:
        return problems       # the rest needs well-formed tables
    # what only the resolved profiles (extends applied) can tell
    for name in profiles:
        where = f"[profiles.{name}]"
        try:
            p = resolve_profile(name, profiles)
        except ConfigError as e:
            problems.append(str(e))
            continue
        missing = [k for k in ("model", "gpus") if not p.get(k)]
        if missing:
            problems.append(f"{where}: needs {' and '.join(missing)}")
        backend = p.get("backend", "vllm")
        ignored = [k for k in OTHER_BACKEND_ARGS[backend] if k in profiles[name]]
        if ignored:
            problems.append(f"{where}: backend '{backend}' ignores {', '.join(ignored)}")
        if d.get("provider") == "vast":
            from . import catalog      # catalog imports this module
            for gpu in p.get("gpus", []):
                try:
                    catalog.provider_gpu(gpu, "vast")
                except ConfigError as e:
                    problems.append(f"{where}: {str(e).splitlines()[0]}")
    default = d.get("profile")
    if default is not None and default not in profiles:
        problems.append(f"[defaults]: profile = '{default}' is not a defined profile "
                        f"(available: {', '.join(profiles)})")
    if d.get("provider") == "vast" and d.get("storage") == "volume":
        problems.append('[defaults]: storage = "volume" is RunPod only (vast.ai has no network volumes)')
    return problems


def load_config(path: Path | None = None) -> Config:
    path = path or project.config
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}\n"
                          "Run `ronin init` here to create one, or pass --config / set RONIN_CONFIG.")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name} is not valid TOML\n{e}") from None
    problems = validate(raw)
    if problems:
        count = f"{len(problems)} problem{'s' if len(problems) > 1 else ''}"
        raise ConfigError(f"{path.name} has {count}\n" + "\n".join(f"- {p}" for p in problems))
    d = raw.get("defaults", {})
    raw_profiles = raw.get("profiles", {})
    profiles = {}
    for name in raw_profiles:
        p = resolve_profile(name, raw_profiles)
        if not p.get("model") or not p.get("gpus"):
            raise ConfigError(f"Profile '{name}' needs both 'model' and 'gpus'")
        backend = p.get("backend", "vllm")
        if backend not in ("vllm", "llamacpp"):
            raise ConfigError(f"Profile '{name}': backend must be 'vllm' or 'llamacpp', "
                              f"not '{backend}'")
        min_cuda = p.get("min_cuda")
        if min_cuda is not None and not re.fullmatch(r"\d+\.\d+", str(min_cuda)):
            raise ConfigError(f"Profile '{name}': min_cuda must look like \"12.8\" "
                              f"(major.minor), not '{min_cuda}'")
        profiles[name] = Profile(name=name, model=p["model"], gpus=list(p["gpus"]),
                                 vllm_args=[str(a) for a in p["vllm_args"]],
                                 disk_gb=int(p.get("disk_gb", 80)),
                                 gpu_count=int(p.get("gpu_count", 1)),
                                 chat_template=p.get("chat_template"),
                                 backend=backend,
                                 llamacpp_args=[str(a) for a in p["llamacpp_args"]],
                                 hf_file=p.get("hf_file"),
                                 image=p.get("image"),
                                 min_cuda=str(min_cuda) if min_cuda is not None else None)
    if not profiles:
        raise ConfigError(f"{path.name} defines no profiles under [profiles.*]")
    storage = d.get("storage", "ephemeral")
    if storage not in ("ephemeral", "volume"):
        raise ConfigError(f"storage must be 'ephemeral' or 'volume', not '{storage}'")
    default_profile = d.get("profile", next(iter(profiles)))
    if default_profile not in profiles:
        raise ConfigError(f"[defaults] profile = '{default_profile}' is not a defined profile\n"
                          f"Available: {', '.join(profiles)}")
    provider = d.get("provider", "runpod")
    check_provider(provider, storage)
    client = d.get("client", "opencode")
    if client not in ("opencode", "claude"):
        raise ConfigError(f"client must be 'opencode' or 'claude', not '{client}'")
    return Config(
        image=d.get("image", "vllm/vllm-openai:v0.32.0"),
        storage=storage,
        datacenter=d.get("datacenter", ""),
        volume_name=d.get("volume_name", "ronin-models"),
        volume_size_gb=int(d.get("volume_size_gb", 120)),
        cloud=d.get("cloud", "SECURE"),
        port=int(d.get("port", 8000)),
        served_name=d.get("served_name", "ronin"),
        client=client,
        opencode_bin=d.get("opencode_bin", "opencode"),
        max_hours=float(d.get("max_hours", 8)),
        idle_minutes=float(d.get("idle_minutes", 60)),
        default_profile=default_profile,
        profiles=profiles,
        provider=provider,
    )


def fallback_config() -> Config:
    """The default settings and no profiles: what `down` / `status` / `logs` use when
    ronin.toml is broken, so a typo never keeps a billing pod out of reach."""
    return Config(image="vllm/vllm-openai:v0.32.0", storage="ephemeral", datacenter="",
                  volume_name="ronin-models", volume_size_gb=120, cloud="SECURE", port=8000,
                  served_name="ronin", client="opencode", opencode_bin="opencode",
                  max_hours=8, idle_minutes=60, default_profile="", profiles={})


PROVIDERS = ("runpod", "vast")


def check_provider(provider: str, storage: str) -> None:
    if provider not in PROVIDERS:
        raise ConfigError(f"provider must be 'runpod' or 'vast', not '{provider}'")
    if provider == "vast" and storage == "volume":
        raise ConfigError('storage = "volume" is RunPod only: vast.ai has no network volumes\n'
                          'Use storage = "ephemeral" with vast (the weights download on each boot).')


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ without overriding existing vars."""
    path = path or project.env
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ConfigError(f"Missing {name}\nAdd it to {project.env} (`ronin init` creates the file).")
    return value


def load_state() -> dict:
    if not project.state.exists():
        return {}
    try:
        return json.loads(project.state.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ConfigError(
            f"State file is unreadable: {project.state}\n"
            f"{e}\n"
            "If a pod is still running, delete it from the provider's console, then remove this file."
        ) from None


def save_state(state: dict) -> None:
    """Write state atomically so an interrupted write cannot corrupt the file."""
    project.state_dir.mkdir(exist_ok=True)
    tmp = project.state.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(project.state)
