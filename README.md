# ronin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Dependencies: none](https://img.shields.io/badge/dependencies-stdlib%20only-brightgreen.svg)](#install)

**Rent a GPU only while you use it.** `ronin` brings a pod serving an LLM up and down
on demand — on [RunPod](https://runpod.io) or [vast.ai](https://vast.ai) — opens a coding
agent connected to it, and makes sure the GPU never keeps billing when you walk away.

```
Your PC ── ronin opencode ──► opencode ──HTTPS──► https://<pod>-8000.proxy.runpod.net    (RunPod)
                                                  https://<random>.trycloudflare.com      (vast.ai)
                                                    └ vLLM or llama-server (/v1, OpenAI-compatible)
                                                      └ /models  ← network volume (RunPod only)
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
- **Two providers.** RunPod or the vast.ai marketplace, chosen by `provider`; profiles
  name GPUs the same way on both.
- **Two backends, one endpoint.** vLLM or llama.cpp behind the same OpenAI `/v1`, selected
  per profile.
- **Standard library only.** No third-party runtime dependencies.

### Contents

[Install](#install) · [Quick start](#quick-start) · [Commands](#commands) ·
[Providers](#providers-runpod-and-vastai) · [Cost guard](#cost-guard) · [Memory check](#memory-check) ·
[Configuration](#configuration-ronintoml) · [Profiles](#profiles) ·
[Console output](#console-output) · [Notes](#notes) · [Tests](#tests)

## Install

```powershell
pip install -e .
ronin init      # creates ronin.toml and .env here if missing; put your RUNPOD_API_KEY (or VAST_API_KEY) in .env
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
ronin prices --hours 3        #    or what each profile would cost now, all in
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
| `ronin doctor` | Check config, API key, account balance, the agent on PATH, stock and whether each profile's model fits its GPUs (on vast.ai also that every GPU is in ronin's table), and pods ronin named on either provider that this project has not registered; exit code 1 on problems |
| `ronin gpus [--min-vram GB] [--secure] [--provider P]` | Available GPUs, price/h and where there is stock: RunPod datacenters, or vast.ai countries (`--secure` lists secure cloud instead of the configured one) |
| `ronin prices [PROFILE...] [--hours H] [--secure] [--provider P]` | What each profile costs now, per GPU in its list: GPU $/h (× `gpu_count`), disk $/h, the weight download (its real size from Hugging Face × the host's $/GB on vast.ai; free on RunPod) and the total for an H-hour session (default 1). Marks the GPU `ronin up` would take (the first with stock); out-of-stock GPUs show their list price, dimmed |
| `ronin profiles` | List the configured profiles (model, GPU, backend, context) |
| `ronin volume [ls\|create\|rm ID [--yes]]` | RunPod network volumes with size and monthly cost; create the configured one; delete one |
| `ronin up [-p profile] [--provider P] [--budget USD] [--max-hours H] [--idle M] [--secure] [--max-wait M] [--keep-on-failure] [--skip-memory-check]` | Check the model fits, create the pod (with the in-pod guard), wait until the server responds. With no stock on the configured cloud it **waits** for it (nothing billed; `--max-wait` caps it); `--secure` uses secure cloud instead of waiting; `--budget 5` makes the guard delete the pod when $5 is spent (GPU + disk + vast.ai's download, priced before creating it) |
| `ronin status [--json]` | State, provider, GPU, cost/h, session cost (estimate + debited so far), guard, URL and account balance, or that a `ronin up` is waiting for stock; warns when the balance runs out before the guard's deadline and about ronin pods not registered here; exit code 0 ready, 2 not ready/gone/waiting for stock, 3 no pod |
| `ronin cost [--days N] [--provider P]` | What the provider actually billed, per pod (named by profile), plus the account balance; flags pods billing has not published yet |
| `ronin logs [-n N] [--no-follow]` | Live container logs (Ctrl+C to exit) |
| `ronin watchdog [--max-hours H] [--poll S]` | Local guard: delete the pod after H hours of uptime |
| `ronin env [--shell powershell\|bash]` | Print the env/config that points the agent at the pod (`ronin env \| Invoke-Expression`) |
| `ronin opencode [--up] [--profile P] [--provider P] [--secure] [--max-wait M] [agent args...]` | Launch the agent (opencode by default; `claude` is an alias) against the pod; everything else is passed to the agent as is |
| `ronin down` | Delete the pod (the volume and weights are kept) |

`ronin opencode --continue`, `ronin opencode -p "explain this repo"` and so on reach the
agent untouched: only ronin's own options (`--up`, `--profile`) are taken, and only
while they come first. Use `--` to pass something that looks like one of them.

`down`, `status`, `logs`, `watchdog` and the agent always act on the provider the
registered pod runs on, even if `provider` in `ronin.toml` changed since `ronin up`.

## Providers: RunPod and vast.ai

`provider` in `[defaults]` (`"runpod"`, the default, or `"vast"`) says where pods run;
`ronin up --provider vast` overrides it for one run. Each needs its key in `.env`
(`RUNPOD_API_KEY`, `VAST_API_KEY`). The profiles do not change: GPUs are named by ronin's
short names (`"RTX 4090"`, `"L40S"`, `"RTX PRO 6000 Server"`) or by RunPod's ids, and an
internal table (`ronin/catalog.py`) translates them for each provider. `cloud` keeps its
meaning: `SECURE` is RunPod's own datacenters / vast.ai's datacenter hosts, `COMMUNITY`
the cheaper hosts.

vast.ai is a marketplace, so ronin also decides internally what a good offer is:
verified hosts with a reliability of 98 % or more (99 % and datacenter-hosted on
`SECURE`), a reachable port, fast download (≥ 500 Mbps) at a low bandwidth price
(≤ $0.02/GB), a CUDA driver the backend supports, enough disk, and a per-GPU price ceiling
(about twice the market's cheapest) so it never rents from a host that overprices. The
cheapest offer that fits is rented; if someone takes it first, the next one.

What differs on vast.ai:

- **Endpoint.** vast.ai has no HTTPS proxy: the port is published as plain HTTP on the
  host's IP. The guard opens a [Cloudflare quick tunnel](https://try.cloudflare.com)
  inside the pod and ronin uses its `https://….trycloudflare.com` URL, so the API key and
  your code are encrypted in transit. If the tunnel does not come up, ronin falls back to
  `http://ip:port` and warns. Quick tunnels are best effort and Cloudflare cuts a request
  that sends nothing for ~100 s, which a very long prefill could hit.
- **No network volumes.** Only `storage = "ephemeral"`: the weights download on every
  boot, and the host may bill the download (`ronin up` and `ronin prices` show what it costs).
- **Logs** are polled (every ~10 s), not streamed.
- **Billing.** `ronin cost` reads vast.ai's per-instance charges (grouped per day).
- **Image pull.** Hosts download the server image themselves, so the first boot on a host
  can take longer (`ronin up` waits up to 30 min for the container).

## Cost guard

Every `ronin up` starts the model server (vLLM or llama-server) through a small wrapper
(`ronin/guard.py`) that runs **inside the pod**, so it works even with your PC asleep or
off. The llama.cpp image has no Python, so there the guard first installs `python3`
(a few seconds at boot); if that fails the pod is deleted rather than left unguarded. It
deletes the pod (using the pod-scoped key the provider puts in the pod: RunPod's
`RUNPOD_API_KEY`, vast.ai's `CONTAINER_API_KEY`) when either:

- the pod has been up for `max_hours` (default 8), or
- the server has served no request for `idle_minutes` (default 60), read from its `/metrics`.

Set both in `[defaults]` of `ronin.toml`, or per run with `ronin up --max-hours 3 --idle 30`;
`0` disables each one. `ronin status` shows when the pod will delete itself, and its
messages appear in `ronin logs` prefixed with `[ronin-guard]`. If the delete call is
refused, the guard falls back to stopping the pod, which also ends GPU billing (on vast.ai
a stopped instance still bills its disk until deleted).

`ronin watchdog` is the local counterpart (it needs your PC on), useful when the guard
is disabled. Both survive network blips: RunPod requests that can be repeated safely
are retried, and the watchdog keeps counting down while RunPod is unreachable.

### What it really costs

The $/h ronin shows and estimates with is the pod's whole hourly bill: the GPU plus its
disk (RunPod's API gives the GPU alone; ronin adds the container disk, ≈ $0.10/GB/month).
A network volume bills monthly on its own.

RunPod bills from the pod's **creation** (image pull and model download included), so
ronin's estimate (`$/h × time since creation`) is labelled as such. The real figures:

- `ronin down` prints the **real cost**: the account balance at `ronin up` minus the
  balance now. It includes anything else billing on the account meanwhile, and the balance
  is debited in batches (RunPod: every ~15 min, the last one after the pod is gone), so a
  short session may not have been debited at all yet. Until the balance drop catches up
  with the estimate, `ronin down` gives the estimate and how much has been debited so far,
  never a misleading `$0.00`.
- `ronin cost` lists what the provider **billed per pod** (the final word), naming each pod by
  its profile. Billing is published with a delay of an hour or more, so a pod deleted
  recently shows up as "not billed yet" with an estimate.
- `ronin status` shows the account balance and what it is spending per hour right now,
  and warns when the balance runs out before the guard's deadline: the provider would
  stop the pod mid-session. It also lists pods named `ronin-*` on the account that this
  project has not registered (another project's, or left by a crashed `up`): they bill
  unseen.
- `ronin up --budget 5` caps a session in dollars instead of hours: the guard's deadline
  becomes the time $5 lasts at the pod's price (GPU + disk; on vast.ai also the weight
  download, and the dearest of the offers `up` may fall back to). GPUs the budget cannot
  keep up for 15 minutes are skipped.

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

- `[defaults]`: provider (`"runpod"` or `"vast"`), vLLM image, storage mode, datacenter
  (RunPod), volume, port and served name.
- `[defaults].cloud`: `"COMMUNITY"` (shipped: cheaper reference prices, ~30–160 % below
  secure, on less reliable hosts, no network volumes) or `"SECURE"`. When the cloud has no
  stock for any of the profile's GPUs, `ronin up` waits and retries every 30 s; `ronin up
  --secure` falls back to secure cloud for that run instead, and says so.
- `[defaults]` also holds the guard limits: `max_hours` and `idle_minutes`.
- `[profiles.<name>]`: `model` (a Hugging Face repo; for llama.cpp add the quant, e.g.
  `org/repo-GGUF:Q4_K_M`), `gpus` (fallback list in order of preference; ronin's short
  names or RunPod ids, see [Providers](#providers-runpod-and-vastai)), optional
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

**Integrity check.** Every command checks `ronin.toml` as a whole before using it and
lists every problem at once: unknown keys (with the likely intended one, or the table a
key belongs in), values of the wrong type or out of range, serving arguments written as
one string (`"--max-model-len 4096"` instead of two items), arguments of the other backend,
unknown `extends`, and, with `provider = "vast"`, GPUs missing from ronin's table. Nothing
is created until the file is sound. `ronin down`, `ronin status` and `ronin logs` still
work with a broken file (on default settings, with a warning), so a typo never keeps a
billing pod out of reach. An unexpected error (a bug) is reported in one line, with its
traceback saved to `.ronin/last-error.log` and a reminder when a pod may be billing.

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
  is served behind RunPod's HTTPS proxy (on vast.ai, a Cloudflare tunnel) and the key
  protects the `/v1` API; `/health` stays
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
