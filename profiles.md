# ronin profiles — recommended matrix (v3)

Recommended model per **tier** (the GPU you can get on RunPod) × **use**
(general / coding / science), from an internal model-comparison audit (v3,
2026-10-21, round-2 audit). This file mirrors what ships in `ronin.toml`.

> **Backends.** ronin now serves with **vLLM** (safetensors / FP8) *or* **llama.cpp**
> (`llama-server`, GGUF), chosen per profile by `backend`. Both expose OpenAI `/v1`, so
> `ronin opencode` is unchanged. v3's primary picks are mostly GGUF; vLLM is used for the
> FP8 picks of `medium` and `huge`.
>
> **Evidence tags.** `[IND]` independent measurement (abliterlitics / satellitedown),
> `[AUT]` the uploader's card (the figure appears there, not that it is true), `[NV]` not
> verified. **TIE-VALIDATE** = no candidate has `[IND]` on both axes (compliance *and*
> capability); decide with your own A/B.
>
> All models are reasoners — set `reasoning_effort` explicitly in the client (opencode);
> some chat templates default it elsewhere. `reasoning_effort` / `max_new_tokens` are
> client settings, not serve flags.

## Tiers → GPU (RunPod community $/h, 2026-10-21)

| Tier | GPU | $/h | RAM pod |
|---|---|---|---|
| **basic** | 1× RTX 4090 24 GB (alt 5090) | 0.34 | ≥32 GB |
| **medium** | 1× L40S 48 GB (Ada: native FP8) | 0.79 | ≥64 GB |
| **big** | 1× RTX PRO 6000 96 GB (Blackwell) | 1.69 | ≥128 GB, disk ≥200 |
| **huge** | 4× H200 (TP4) | 14.36 | ≥500 GB, disk ≥600 |

## The matrix

| Profile | Backend | Model (HF) | Served | Notes |
|---|---|---|---|---|
| **basic-general** | llama.cpp | `bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF:Q4_K_M` | 17.77 GB | Ranking #1 (82.2 %). `[IND]` harmful-only 96.7 %, delivered 71.0 %, drift 0.48 pp (lowest of 12). Text-only (`--no-mmproj`). |
| **basic-coding** | llama.cpp | `mradermacher/…Coder390-EfficientThink-abliterated-i1-GGUF:Q4_K_M` | ~17 GB | **TIE-VALIDATE**. User's pick (2026-10-21) vs orcarouter (`default_until_ab`). MTP head. |
| **basic-science** | llama.cpp | orcarouter Q4_K_M (Q6_K on a 5090) | 17.77 GB | Ties the base in science (`[IND]` GPQA −1.0, within noise). KV q8_0. |
| **medium-general** | vLLM | `id-2/Qwen3.8-27B-Uncensored-FP8` (verified mirror) | 30.87 GB | Ada native FP8. Do **not** pass `--quantization fp8`. XML tool calls → `qwen3_coder` parser; qwen3 reasoning; vision inputs off. |
| **medium-coding** | llama.cpp | `mradermacher/…Coder390-EfficientThink-abliterated-GGUF:Q6_K` (static repo, MTP) | 22.43 GB | **TIE-VALIDATE** (same A/B as basic-coding). 128K, KV f16 (192K looped above ~140K). Benchmarked +45–55 % decode over Q8_0 (59.7 vs 38.6 tok/s at 41K). |
| **medium-science** | vLLM | orcarouter FP8 | 30.87 GB | = medium-general, longer reasoning budget (client xhigh). |
| **big-general** | llama.cpp | `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF:IQ3_S` | ~51 GiB VRAM | 125B MoE / 6B active. `[AUT]` GPQA 92.9, LCB 86.9. compress_ratios correct (alt GGUFs ship zeros); n-gram table on host via `-lm mmap`. |
| **big-coding** | llama.cpp | = big-general | — | base DeepSWE 58.7, SWE-bench Pro 62.5. reasoning medium. |
| **big-science** | llama.cpp | = big-general, KV q8_0 | — | reasoning xhigh. `[NV]` loops at xhigh on this build. |
| **huge-general** | vLLM | `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` | 328.3 GB | 4× H200 TP4. `[AUT]` MMLU 87.33, 0 % refusals HarmBench-320. Adjust GLM parser names to your vLLM build. |
| **huge-coding** | *SGLang (pending)* | `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` | 510.3 GB | **Not servable yet** — needs SGLang (engram host table, TP4+EP4). Commented in `ronin.toml`. base DeepSWE **77.0**. |
| **huge-science** | *SGLang (pending)* | = huge-coding | — | **Not servable yet**. HLE 42.6. |

## Mapping from the retired `ronin.toml`

The old eval-control profiles (`main`, `main-reasoning`, `plain`, `base`, `big`, `huge`)
were **retired**; `ronin.toml` is now only the v3 matrix. `[defaults].profile` is
`basic-general`. If you still want the degradation baseline for `eval/`, re-add a vLLM
`Qwen/Qwen3.8-27B` profile by hand.

## Caveats carried into `ronin.toml`

- **big-\***: SC117's two-shard GGUF, n-gram table on the host via `-lm mmap`. Its
  `compress_ratios` are correct (`[0,0,0,4,…]`, read from the header); only the
  alternative GGUFs ship zeros. The MTP draft (`spiritfather` Q8_0) is left commented.
- **huge-coding / huge-science**: need an SGLang backend ronin does not serve yet →
  shipped commented out. Everything else is live.

See the per-profile comments in `ronin.toml` for each TIE-VALIDATE's alternatives
(`alt_low_drift`, `alt_conservative`, `alt_fast`, …) and the exact A/B protocol.
