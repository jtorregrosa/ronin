# CLAUDE.md

## What this project is

`ronin` is a small Python CLI (standard library only, Python 3.11+) that manages the
lifecycle of a cloud GPU pod on [RunPod](https://runpod.io) or [vast.ai](https://vast.ai)
(chosen by `provider` in `ronin.toml`) which serves a model —
with [vLLM](https://github.com/vllm-project/vllm) (safetensors/FP8) or
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` (GGUF), chosen by
`backend` in each profile — and launches a coding agent (opencode by default, or Claude
Code) pointed at that pod's endpoint. The agent is chosen by `client` in `ronin.toml`:
opencode uses the OpenAI `/v1` that both backends expose, Claude Code vLLM's Anthropic
`/v1/messages`.

It is an infrastructure / developer-tooling project: a thin REST client, a CLI, config
and local-state handling, and console UI. The work here is ordinary software
engineering — CLI design, HTTP, process management, and terminal output.

```
Your PC ── ronin opencode ──► opencode ──HTTPS──► https://<pod>-8000.proxy.runpod.net
                                                    └ vLLM or llama-server
                                                      (/v1 OpenAI; vLLM also /v1/messages for Claude Code)
```

## Layout

- `ronin/cli.py` — argparse CLI and the commands (`init`, `doctor`, `gpus`, `prices`, `profiles`,
  `volume`, `up`, `down`, `status`, `cost`, `logs`, `env`, `opencode` — with `claude` as an alias).
- `ronin/pod.py` — shared pod plumbing: provider client (`client(provider)`; the
  registered pod's provider is in the state), pod definition (`server_spec` builds the
  server from a per-backend builder in `BACKENDS` — `_vllm_spec` / `_llamacpp_spec` —
  and `pod_body` / `vast_body` wrap it for each provider), endpoint URL, health check, log
  stream, termination, and the client config (`opencode_config`, `claude_env`) that points
  the agent at the pod. `serve_arg` / `max_model_len` read whichever args the backend uses.
- `ronin/watchdog.py` — `ronin watchdog`, the local uptime guard.
- `ronin/guard.py` — runs *inside the pod* (sent as `python3 -c`): wraps the server
  (`vllm serve` or `llama-server`) and deletes the pod at a deadline or when idle (through
  the provider's API, chosen by `RONIN_PROVIDER`); on vast.ai it also opens a Cloudflare
  quick tunnel for HTTPS. Reads the backend's Prometheus metric names from the env.
  Standard library only; ronin itself never imports it.
- `ronin/fit.py` — memory check: model weight size (Hugging Face API) against each
  GPU's VRAM, run by `up` and `doctor`; and the download size of a profile (its files'
  sizes on Hugging Face), priced by `prices`.
- `ronin/api.py` — what the provider clients share: `ProviderError` and a JSON client
  that retries idempotent requests on transient errors.
- `ronin/runpod.py` — minimal client for the RunPod REST API v2, plus the account balance
  (GraphQL) and per-pod billing (REST v1, published with a delay of an hour or more).
- `ronin/vast.py` — minimal client for the vast.ai REST API (offer search + rent, logs by
  polling). It returns pods in the RunPod client's shape, so commands work on either.
- `ronin/catalog.py` — internal normalisation tables: GPU names per provider (with VRAM
  and a vast.ai price ceiling), vast.ai offer filters per `cloud`, vast.ai statuses.
- `ronin/config.py` — project discovery, `ronin.toml` parsing (profiles with `extends`),
  `.env` loading, local state in `.ronin/` next to `ronin.toml`.
- `tests/` — `python -m unittest`.
- `ronin/ui.py` — console formatting (colours, symbols, progress line).
- `ronin/mascot.py` — ASCII mascot shown by some commands.
- `templates/` — vLLM chat templates.

## Conventions

- Standard library only; no third-party runtime dependencies.
- All console output goes through `ronin/ui.py` for a consistent style; respect
  `NO_COLOR`, non-TTY output, and ASCII fallbacks.
- Costs matter: commands that leave a GPU pod running should make the cost and how to
  stop it obvious (`ronin down`, `ronin status` warnings, `ronin watchdog`).
- User-facing docs are in English (`README.md`); keep the command table there in sync
  when you add or change a command.

## Working notes

- The user often writes in Spanish; reply in the language they use.
- Target OS is Windows (PowerShell is the primary shell), so prefer cross-platform code
  and test PowerShell-facing output paths.
