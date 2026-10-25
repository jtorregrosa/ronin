# ronin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Dependencies: none](https://img.shields.io/badge/dependencies-stdlib%20only-brightgreen.svg)](#install)

**Rent a GPU only while you use it.** `ronin` brings a [RunPod](https://runpod.io) pod
serving an LLM up and down on demand, opens a coding agent connected to it, and makes
sure the GPU never keeps billing when you walk away.

```
Your PC ── ronin opencode ──► opencode ──HTTPS──► https://<pod>-8000.proxy.runpod.net
                                                    └ vLLM or llama-server (/v1, OpenAI-compatible)
                                                      └ /models  ← network volume (persistent weights)
```

The pod serves with **vLLM** (safetensors / FP8) or **llama.cpp**'s `llama-server`
(GGUF), chosen per profile by `backend`. The agent is set by `client` in `ronin.toml`
(`"opencode"` or `"claude"`): opencode talks to the OpenAI `/v1` endpoint both backends
expose; Claude Code uses vLLM's Anthropic-compatible `/v1/messages` (so it needs a vLLM
profile).

### Why ronin

- **No idle spend.** A guard runs *inside* the pod and deletes it after a time limit or
  when it goes idle — even with your PC asleep or off. `ronin down` ends a session at once.
- **One command to work.** `ronin opencode` sizes the model, brings the pod up, waits for
  the server and launches the agent against it.
- **Honest costs.** ronin separates its running estimate from what RunPod actually billed,
  and flows the agent's args through untouched.
- **Two backends, one endpoint.** vLLM or llama.cpp behind the same OpenAI `/v1`, selected
  per profile.
- **Standard library only.** No third-party runtime dependencies.

### Contents

[Install](#install) · [Quick start](#quick-start) · [Commands](#commands) ·
[Cost guard](#cost-guard) · [Memory check](#memory-check) ·
[Configuration](#configuration-ronintoml) · [Profiles](#profiles) ·
[Console output](#console-output) · [Notes](#notes) · [Tests](#tests)

## Install

```powershell
pip install -e .
ronin init      # creates ronin.toml and .env here if missing; put your RUNPOD_API_KEY in .env
ronin doctor    # checks config, API key, the agent on PATH, GPU stock and model memory
```

Requires Python 3.11+ and has no external dependencies.

**Where ronin looks for its files.** The project is the directory holding `ronin.toml`,
found from (in order) `--config`, the `RONIN_CONFIG` variable, the current directory and
its parents, and finally the checkout ronin is installed from. `.env` and the local state
(`.ronin/`) live next to it, so you can run `ronin opencode` from any repository and still
use this setup.

## Quick start

```powershell
ronin gpus --min-vram 48      # 1. see which GPUs have stock, their price and datacenters
ronin up                      # 2. bring up the default profile (basic-general)
ronin opencode                # 3. open the agent against the pod
ronin down                    # 4. delete the pod and stop paying for the GPU
```

`ronin up -p big-coding` picks another profile. `ronin opencode --up` does steps 2–3 in
one go. If the chosen cloud has no stock, `ronin up` **waits** for it (nothing is billed
while waiting); `--secure` uses the secure cloud instead of waiting.

**Pick a storage mode** in `ronin.toml` (`storage`) before your first run:

- `"ephemeral"` (default): no fixed cost. Each `ronin up` downloads the model (a few
  minutes) and `ronin down` deletes it. Runs in any datacenter with stock.
- `"volume"`: a persistent network volume (~$0.07/GB/month). Fast boots, but the pod is
  pinned to the volume's datacenter. Set `datacenter` (see `ronin gpus --min-vram 48`) and
  create the volume once with `ronin volume create`.

## Commands

| Command | What it does |
|---|---|
| `ronin init` | Set up a project here: `ronin.toml`, `.env`, `.gitignore` entries (and the volume in volume mode) |
| `ronin doctor` | Check config, API key, the agent on PATH, stock and whether each profile's model fits its GPUs; exit code 1 on problems |
| `ronin gpus [--min-vram GB]` | Available GPUs, price/h and datacenters with stock |
| `ronin profiles` | List the configured profiles (model, GPU, backend, context) |
| `ronin volume [ls\|create\|rm ID [--yes]]` | Network volumes with size and monthly cost; create the configured one; delete one |
| `ronin up [-p profile] [--max-hours H] [--idle M] [--secure] [--max-wait M] [--keep-on-failure] [--skip-memory-check]` | Check the model fits, create the pod (with the in-pod guard), wait until the server responds. With no stock on the configured cloud it **waits** for it (nothing billed; `--max-wait` caps it); `--secure` uses secure cloud instead of waiting |
| `ronin status [--json]` | State, GPU, cost/h, session cost (estimate + real so far), guard, URL and account balance, or that a `ronin up` is waiting for stock; exit code 0 ready, 2 not ready/gone/waiting for stock, 3 no pod |
| `ronin cost [--days N]` | What RunPod actually billed, per pod (named by profile), plus the account balance; flags pods billing has not published yet |
| `ronin logs [-n N] [--no-follow]` | Live container logs (Ctrl+C to exit) |
| `ronin watchdog [--max-hours H] [--poll S]` | Local guard: delete the pod after H hours of uptime |
| `ronin env [--shell powershell\|bash]` | Print the env/config that points the agent at the pod (`ronin env \| Invoke-Expression`) |
| `ronin opencode [--up] [--profile P] [--secure] [--max-wait M] [agent args...]` | Launch the agent (opencode by default; `claude` is an alias) against the pod; everything else is passed to the agent as is |
| `ronin down` | Delete the pod (the volume and weights are kept) |

`ronin opencode --continue`, `ronin opencode -p "explain this repo"` and so on reach the
agent untouched: only ronin's own options (`--up`, `--profile`) are taken, and only
while they come first. Use `--` to pass something that looks like one of them.

## Cost guard

Every `ronin up` starts the model server (vLLM or llama-server) through a small wrapper
(`ronin/guard.py`) that runs **inside the pod**, so it works even with your PC asleep or
off. The llama.cpp image has no Python, so there the guard first installs `python3`
(a few seconds at boot); if that fails the pod is deleted rather than left unguarded. It
deletes the pod (using the pod-scoped `RUNPOD_API_KEY` RunPod provides) when either:

- the pod has been up for `max_hours` (default 8), or
- the server has served no request for `idle_minutes` (default 60), read from its `/metrics`.

Set both in `[defaults]` of `ronin.toml`, or per run with `ronin up --max-hours 3 --idle 30`;
`0` disables each one. `ronin status` shows when the pod will delete itself, and its
messages appear in `ronin logs` prefixed with `[ronin-guard]`. If the delete call is
refused, the guard falls back to stopping the pod, which also ends GPU billing.

`ronin watchdog` is the local counterpart (it needs your PC on), useful when the guard
is disabled. Both survive network blips: RunPod requests that can be repeated safely
are retried, and the watchdog keeps counting down while RunPod is unreachable.

### What it really costs

RunPod bills from the pod's **creation** (image pull and model download included), so
ronin's estimate (`$/h × time since creation`) is labelled as such. The real figures:

- `ronin down` prints the **real cost so far**: the account balance at `ronin up` minus
  the balance now. It is available at once, but includes anything else billing on the
  account meanwhile, and RunPod debits the last few minutes after the pod is gone (once
  about half the total), so when the balance still trails the estimate `ronin down` says
  so and shows both.
- `ronin cost` lists what RunPod **billed per pod** (the final word), naming each pod by
  its profile. Billing is published with a delay of an hour or more, so a pod deleted
  recently shows up as "not billed yet" with an estimate.
- `ronin status` shows the account balance and what it is spending per hour right now.

## Memory check

Before creating a pod, `ronin up` sizes the model's weights from the Hugging Face API
(the safetensors parameter count, no download) and compares them with each GPU of the
profile: weights plus ~3 GiB for vLLM must fit in VRAM × `--gpu-memory-utilization`.
GPUs that are too small are skipped; if none is big enough, `ronin up` stops before
anything is billed and suggests fp8 or a larger GPU. Online quantization
(`--quantization fp8*`) counts as 1 byte per parameter; pre-quantized checkpoints
(FP8, AWQ, GPTQ) count their own size. It also warns when little memory is left for
the KV cache. It is an estimate: `--skip-memory-check` tries anyway, and if Hugging Face
cannot be reached (or a gated model needs `HF_TOKEN`) the check is skipped with a
warning. `ronin doctor` runs the same check for every profile. **llama.cpp (GGUF)
profiles skip this check**: GGUF repos publish no safetensors metadata, so size the quant
against your GPU's VRAM yourself (each quant's size is listed on its Hugging Face page).

## Configuration (`ronin.toml`)

- `[defaults]`: vLLM image, storage mode, datacenter, volume, port and served name.
- `[defaults].cloud`: `"COMMUNITY"` (shipped: cheaper reference prices, ~30–160 % below
  secure, on less reliable hosts, no network volumes) or `"SECURE"`. When the cloud has no
  stock for any of the profile's GPUs, `ronin up` waits and retries every 30 s; `ronin up
  --secure` falls back to secure cloud for that run instead, and says so.
- `[defaults]` also holds the guard limits: `max_hours` and `idle_minutes`.
- `[profiles.<name>]`: `model` (a Hugging Face repo; for llama.cpp add the quant, e.g.
  `org/repo-GGUF:Q4_K_M`), `gpus` (fallback list in order of preference), optional
  `disk_gb`, `gpu_count` (GPUs per pod; >1 serves one model across them with tensor
  parallelism, at that many times the cost), `image` (override the backend's default) and
  `min_cuda` (e.g. `"12.8"`: only hosts whose driver supports that CUDA version; llama.cpp
  profiles default to 12.8, what its image requires).
  - **vLLM profiles** (default, `backend = "vllm"`): `vllm_args`, `chat_template`
    (relative to `ronin.toml`, passed inline). `extra_vllm_args` appends to the inherited
    `vllm_args`.
  - **llama.cpp profiles** (`backend = "llamacpp"`): `llamacpp_args` (e.g.
    `["-c", "65536", "-ngl", "99", "-fa", "on"]`), optional `hf_file` (`-hff`, an exact
    GGUF filename). `extra_llamacpp_args` appends to the inherited ones. ronin always adds
    `--jinja` (OpenAI tool-calling) and `--metrics` (for the idle guard).
  - `extends = "<profile>"` reuses another profile and overrides only the keys it sets.

To switch models, edit the profile's `model`, or add a new profile.

## Profiles

The shipped `ronin.toml` is a model-comparison matrix of **tier** (the GPU you can get on
RunPod) × **use** (general / coding / science). Most tiers serve GGUF with llama.cpp; the
FP8 picks of `medium` and `huge` serve with vLLM. The models are *abliterated/uncensored*
variants (refusal behaviour removed) and all are reasoners — set `reasoning_effort`
explicitly in your client (opencode), since some chat templates default it elsewhere.

| Profile | Backend | GPU tier | Model | Context |
|---|---|---|---|---|
| **basic-general** *(default)* | llama.cpp | RTX 4090 24 GB | `orcarouter/Qwen3.8-27B-Uncensored` (GGUF Q4_K_M) | 64K |
| **basic-coding** | llama.cpp | RTX 4090 | huihui `Coder390-EfficientThink` (GGUF Q4_K_M, MTP) | 48K |
| **basic-science** | llama.cpp | RTX 4090 | orcarouter Q4_K_M, KV q8_0 | 64K |
| **medium-general** | vLLM | L40S 48 GB | `id-2/Qwen3.8-27B-Uncensored-FP8` | 128K |
| **medium-coding** | llama.cpp | L40S | huihui Coder390 (GGUF Q6_K, MTP) | 128K |
| **medium-science** | vLLM | L40S | orcarouter FP8 | 128K |
| **big-general** | llama.cpp | RTX PRO 6000 96 GB | `SC117/Qwen3.8-Flash-Next-GSQ-RCO` (GGUF IQ3_S) | 192K |
| **big-coding** | llama.cpp | RTX PRO 6000 | = big-general | 192K |
| **big-science** | llama.cpp | RTX PRO 6000 | = big-general, KV q8_0 | 192K |
| **huge-general** | vLLM | 4× H200 (TP4) | `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` | 128K |

Several coding picks are a **tie** (no candidate wins on both compliance and capability):
the shipped choice is a default pending an A/B — see the per-profile comments in
`ronin.toml` for the alternatives, evidence tags and the A/B to run.

Two caveats, documented in `ronin.toml`:

- **`big-*`** serves SC117's two-shard Flash-Next GGUF; its 28.8 GB n-gram table stays
  on the host (`-lm mmap`). SC117's build has correct `compress_ratios` (the alternative
  GGUFs are the ones with zeros). The MTP draft is left commented: SC117's own is in the
  old layout.
- **`huge-coding` / `huge-science`** (DeepSeek-V4.1-Flash, a strong open coder/science
  base) need an **SGLang** backend that ronin does not serve yet, so they ship commented
  out in `ronin.toml`.

All profiles use `storage = "ephemeral"`, so each `ronin up` downloads the weights and
`ronin down` discards them (no fixed volume cost, slower boots). See `ronin profiles` for
the terse model/GPU/backend/context listing.

### Keeping the matrix current

New abliterated models appear on Hugging Face every day. `scout/` is a standalone
helper that finds them, checks which tier each fits, asks Claude (`claude -p`) for a
verdict against the baseline study the matrix came from, and writes a dated report in
`scout/reports/` with profiles to paste and the `eval/` A/B commands to validate them.
It never edits `ronin.toml` or starts a pod.

```powershell
python scout/run.py              # full run (needs Claude Code on PATH for the verdict)
python scout/run.py --no-judge   # without Claude
```

See [`scout/README.md`](scout/README.md).

## Console output

- Steps `›`, success `✓`, warnings `!` and errors `✗`, with subtle colours (commands
  in cyan, detail dimmed). No colour when `NO_COLOR` is set or the output is not a
  terminal; ASCII symbols when the console cannot render the Unicode ones.
- `ronin up` shows the real startup phase (image, weights, compilation, CUDA graphs…)
  on a single line with elapsed time.
- `ronin status` and `ronin down` show the session duration and approximate cost, and
  `ronin status` warns when a pod has been billing for a long time.
- `ronin watchdog` runs in the foreground (leave it in a spare terminal) showing a live
  countdown, and deletes the pod once its total uptime reaches the limit. The limit
  counts the pod's uptime, not the watchdog's, so restarting it does not reset the clock.
- A kasa-wearing samurai mascot accompanies `up`, `down` and `status`, with a pose per
  state (ready, waiting, failure, down, no pod). Disable it with `RONIN_NO_MASCOT=1`.

## Notes

- `ronin down` always deletes the pod: no GPU cost between sessions. In ephemeral mode
  nothing is stored either; with a volume you only pay for the volume.
- Ephemeral cost: the download minutes are billed at GPU rate (~$0.05–0.20 per boot on
  an L40S). It beats a volume unless you boot many dozens of times a month.
- Each `ronin up` mints a fresh API key, stored in `.ronin/state.json` (gitignored) and
  used automatically by `ronin opencode`. It reaches the server through an env var, not
  the command line (`VLLM_API_KEY` for vLLM, `LLAMA_API_KEY` for llama.cpp). The endpoint
  is served behind RunPod's HTTPS proxy and the key protects the `/v1` API; `/health` stays
  open. `/metrics` is open on vLLM but key-protected on llama-server (the guard sends the
  pod's key).
- A stopped or crashed pod is deleted rather than forgotten, and if a datacenter change
  leaves an old volume behind, `ronin` says so (`ronin volume ls` shows what each costs).

## Tests

```powershell
python -m unittest
```

For example, `.env` precedence: a variable already present in the real environment wins
over `.env` (it is loaded with `setdefault`).
