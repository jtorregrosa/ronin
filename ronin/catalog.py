"""What ronin normalises between providers, so ronin.toml only says what the user wants.

GPUs: a profile's `gpus` may name a GPU by ronin's short name ("RTX 4090") or by its
RunPod id ("NVIDIA GeForce RTX 4090"); the table below gives each provider's own name.
The vast.ai names, VRAM and reference prices were read from vast's offer search
(2026-10-27). `vast_max_dph` is a per-GPU ceiling in $/h -- roughly twice the cheapest
verified offer -- so ronin never rents from a host that prices far above the market.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ConfigError


@dataclass(frozen=True)
class Gpu:
    name: str                 # ronin's short name
    runpod: str | None        # RunPod's gpu id (None: not offered there)
    vast: str | None          # vast.ai's gpu_name (None: not offered there)
    vram_gb: int              # nominal per-card memory
    vast_max_dph: float = 0   # $/h per card on vast; 0 = no ceiling


GPUS = [Gpu(*row) for row in (
    # name                  RunPod id                                            vast gpu_name         GB   $/h max
    ("RTX 3090",            "NVIDIA GeForce RTX 3090",                           "RTX 3090",           24, 0.50),
    ("RTX 4090",            "NVIDIA GeForce RTX 4090",                           "RTX 4090",           24, 0.90),
    ("RTX 5090",            "NVIDIA GeForce RTX 5090",                           "RTX 5090",           32, 1.10),
    ("RTX A5000",           "NVIDIA RTX A5000",                                  "RTX A5000",          24, 0.50),
    ("RTX A6000",           "NVIDIA RTX A6000",                                  "RTX A6000",          48, 0.90),
    ("RTX 6000 Ada",        "NVIDIA RTX 6000 Ada Generation",                    "RTX 6000Ada",        48, 1.20),
    ("L40",                 "NVIDIA L40",                                        "L40",                48, 1.00),
    ("L40S",                "NVIDIA L40S",                                       "L40S",               48, 1.50),
    ("RTX PRO 6000 Server", "NVIDIA RTX PRO 6000 Blackwell Server Edition",      "RTX PRO 6000 S",     96, 3.00),
    ("RTX PRO 6000 WS",     "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000 WS",    96, 3.00),
    ("RTX PRO 6000 Max-Q",  None,                                                "RTX PRO 6000 Max-Q", 96, 2.50),
    ("A100 PCIe 80GB",      "NVIDIA A100 80GB PCIe",                             "A100 PCIE",          80, 2.00),
    ("A100 SXM 80GB",       "NVIDIA A100-SXM4-80GB",                             "A100 SXM4",          80, 2.00),
    ("H100 PCIe",           "NVIDIA H100 PCIe",                                  "H100 PCIE",          80, 3.50),
    ("H100 SXM",            "NVIDIA H100 80GB HBM3",                             "H100 SXM",           80, 4.00),
    ("H100 NVL",            "NVIDIA H100 NVL",                                   "H100 NVL",           94, 4.50),
    ("H200",                "NVIDIA H200",                                       "H200",              141, 6.00),
    ("H200 NVL",            "NVIDIA H200 NVL",                                   "H200 NVL",          141, 6.00),
    ("B200",                "NVIDIA B200",                                       "B200",              180, 12.00),
)]


# Every spelling ronin accepts, lower-cased -> the row.
_BY_ALIAS = {alias.lower(): g for g in GPUS for alias in (g.name, g.runpod, g.vast) if alias}


def find(gpu: str) -> Gpu | None:
    return _BY_ALIAS.get(gpu.strip().lower())


def key(gpu: str) -> str:
    """One name per GPU whatever the spelling (unknown GPUs are their own key)."""
    row = find(gpu)
    return row.name if row else gpu


def offer_row(gpu_name: str, ram_mib: float | None) -> Gpu | None:
    """The row of a vast.ai offer or instance. vast uses one name for several memory sizes
    (A100 PCIE: 40 and 80 GB), so the row only counts when the memory matches it."""
    row = find(gpu_name)
    if row and (ram_mib or 0) >= min_ram_mib(row):
        return row
    return None


def offer_key(gpu_name: str, ram_mib: float | None) -> str:
    row = offer_row(gpu_name, ram_mib)
    return row.name if row else gpu_name


def min_ram_mib(row: Gpu) -> float:
    """vast's gpu_ram (MiB, as CUDA reports it) of a card of this row, with headroom."""
    return row.vram_gb * 1000 * 0.9


def provider_gpu(gpu: str, provider: str) -> str:
    """The provider's own name for `gpu`. RunPod gets unknown names untouched (a GPU it
    added after this table); vast.ai never gets a guessed one: a wrong name rents the
    wrong card, so an unknown GPU is an error."""
    row = find(gpu)
    name = getattr(row, provider) if row else None
    if name:
        return name
    if provider == "runpod":
        return gpu
    known = ", ".join(g.name for g in GPUS if getattr(g, provider))
    if row:
        raise ConfigError(f"GPU '{gpu}' is not offered on {provider}\nGPUs ronin knows there: {known}")
    raise ConfigError(f"GPU '{gpu}' is not in ronin's GPU table, so it cannot be rented on {provider}\n"
                      f"GPUs ronin knows: {known}")


def vram_gb(gpu: str) -> int | None:
    row = find(gpu)
    return row.vram_gb if row else None


# vast.ai offer filters behind each `cloud` value. COMMUNITY is any verified host with a
# good track record; SECURE is vast's datacenter hosts (hosting_type 1), stricter still.
VAST_CLOUDS = {
    "COMMUNITY": {"verified": {"eq": True}, "reliability2": {"gte": 0.98}},
    "SECURE": {"verified": {"eq": True}, "reliability2": {"gte": 0.99}, "hosting_type": {"eq": 1}},
}


# Filters applied to every vast.ai search, whatever the cloud.
VAST_BASE_FILTERS = {
    "rentable": {"eq": True},
    "rented": {"eq": False},
    "external": {"eq": False},
    "direct_port_count": {"gte": 1},      # the model port must be reachable
    "cpu_arch": {"eq": "amd64"},          # the server images (and cloudflared) are x86
    "inet_down": {"gte": 500},            # Mbps: the weights are downloaded on every boot
    "inet_down_cost": {"lte": 0.02},      # $/GB: some hosts bill the download dearly
}


# vast.ai's actual_status -> the RunPod-style status the rest of ronin reads.
VAST_STATUS = {
    "running": "RUNNING",
    "exited": "EXITED",
    "offline": "OFFLINE",
    "unknown": "UNKNOWN",
    "created": "CREATED",
    "loading": "CREATED",
    None: "CREATED",
}
