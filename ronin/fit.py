"""Memory check: will the model's weights fit on each GPU of a profile?

The weight size comes from the Hugging Face API (the safetensors metadata: parameter
count per dtype), so the check needs no download. It is an estimate: it catches the
"BF16 80B model on a 96 GB card" mistake before a pod is billed, not the last GiB.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

from . import USER_AGENT
from .pod import serve_arg

HF_API = "https://huggingface.co/api/models/{}?expand%5B%5D=safetensors"
GIB = 1024 ** 3
# Bytes per parameter as Hugging Face counts them. Packed int4 checkpoints (GPTQ, AWQ)
# report I32 tensors in unpacked parameters, so each one is half a byte.
DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "BF16": 2, "F16": 2, "I16": 2,
               "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1, "I32": 0.5}
# Quantization vLLM applies while loading a 16/32-bit checkpoint: bytes per parameter.
ONLINE_QUANT = {"fp8": 1, "bitsandbytes": 0.5}
DEFAULT_UTILIZATION = 0.9        # vLLM's --gpu-memory-utilization default
# RunPod lists VRAM in round "GB" that is slightly above the GiB CUDA reports
# ("96 GB" RTX PRO 6000 = 94.97 GiB, "80 GB" H100 = 79.2 GiB).
VRAM_GIB_PER_GB = 0.99
# Activations, CUDA graphs and non-PyTorch memory on top of the weights.
OVERHEAD_GIB = 3
# Below this much room for the KV cache, long contexts are unlikely to fit.
TIGHT_GIB = 4


@lru_cache(maxsize=None)  # doctor asks once per profile, often for the same model
def checkpoint_params(model: str, timeout: float = 10) -> dict[str, int]:
    """Parameters per dtype from the Hugging Face API. Raises LookupError when the
    size cannot be known (offline, gated without HF_TOKEN, no safetensors)."""
    headers = {"User-Agent": USER_AGENT}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    url = HF_API.format(urllib.parse.quote(model, safe="/"))
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=timeout) as resp:
            info = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise LookupError(f"Hugging Face answered {e.code} for {model}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
        raise LookupError(f"could not reach Hugging Face ({e})") from None
    params = (info.get("safetensors") or {}).get("parameters")
    if not params:
        raise LookupError(f"{model} publishes no safetensors size")
    return params


def online_quant(profile) -> tuple[str, float] | None:
    """(method, bytes per parameter) when vLLM quantizes the weights on load."""
    method = serve_arg(profile, "--quantization", "-q")
    if not method:
        return None
    for prefix, size in ONLINE_QUANT.items():
        if method.startswith(prefix):
            return method, size
    return None  # a pre-quantized checkpoint (awq, gptq...): its own size already counts


def weights_gib(params: dict[str, int], profile) -> float:
    quant = online_quant(profile)
    total = 0.0
    for dtype, count in params.items():
        size = DTYPE_BYTES.get(dtype, 2)
        if quant and size >= 2:
            size = quant[1]
        total += count * size
    return total / GIB


def precision(params: dict[str, int], profile) -> str:
    """How the weights are held, for messages: 'BF16', 'fp8_per_tensor on load'..."""
    quant = online_quant(profile)
    if quant:
        return f"{quant[0]} on load"
    dtype = max(params, key=params.get)
    return {"F8_E4M3": "FP8", "F8_E5M2": "FP8", "I32": "int4"}.get(dtype, dtype)


@dataclass
class Fit:
    gpu: str
    vram_gb: float | None   # None: the GPU is not in RunPod's catalog
    spare_gib: float | None  # room left for the KV cache (negative: does not fit)

    @property
    def fits(self) -> bool:
        return self.spare_gib is None or self.spare_gib > 0

    @property
    def tight(self) -> bool:
        return self.spare_gib is not None and 0 < self.spare_gib < TIGHT_GIB


def fits(profile, weights: float, catalog: list[dict]) -> list[Fit]:
    """One Fit per GPU of the profile, in the profile's order. With gpu_count > 1 the
    weights are spread over that many cards (tensor parallelism), so the pooled VRAM is
    what must hold them; vram_gb is reported per card."""
    vram = {g["id"]: g.get("memory") for g in catalog}
    utilization = float(serve_arg(profile, "--gpu-memory-utilization") or DEFAULT_UTILIZATION)
    count = max(1, getattr(profile, "gpu_count", 1))
    result = []
    for gpu in profile.gpus:
        gb = vram.get(gpu)
        spare = None
        if gb:
            spare = gb * count * VRAM_GIB_PER_GB * utilization - weights - OVERHEAD_GIB
        result.append(Fit(gpu, gb, spare))
    return result


def too_small_hint(profile, params: dict[str, int]) -> str:
    if online_quant(profile):
        return "Use a GPU with more memory (`ronin gpus`) or a smaller model."
    return ('Add "--quantization", "fp8_per_tensor" to the profile\'s vllm_args (about half '
            "the memory), use a quantized checkpoint, or a GPU with more memory (`ronin gpus`).")
