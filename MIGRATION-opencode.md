# Migration plan: Claude Code → opencode

Replace the client ronin launches (Claude Code) with
[opencode](https://opencode.ai), a provider-agnostic terminal coding agent.

## Why

ronin exists to point a coding agent at a self-hosted pod. Claude Code only
speaks the **Anthropic `/v1/messages`** protocol, which forces the pod to expose
an Anthropic-compatible endpoint (today: vLLM's experimental one) and makes any
non-vLLM backend need a translation proxy.

opencode talks to any **OpenAI-compatible `/v1/chat/completions`** endpoint via
the `@ai-sdk/openai-compatible` package. That:

- removes the Anthropic constraint and the need for any proxy/shim;
- lets the pod keep serving with vLLM **unchanged** (vLLM already serves OpenAI
  `/v1` on the same port as the Anthropic one);
- turns a future switch to **llama.cpp / Ollama / TabbyAPI** (to serve GGUF or
  exotic quants) into a `baseURL` change instead of a new serving stack.

## Scope

- **Changes:** the client-facing parts of ronin — the `claude` command, the
  env/config it generates, the `doctor` check, and docs.
- **Unchanged:** everything that runs the pod — `pod_body`, the guard, vLLM args,
  profiles, health check, volume/state handling. The pod does not change.

The migration is split so the risky backend work is **not** entangled with the
client swap.

---

## Phase 1 — opencode over the existing vLLM pod

Goal: `ronin opencode` opens opencode against the current pod, same vLLM, no
proxy. This is the whole migration for day-to-day use.

### 1. `config.py`

Add two optional `[defaults]` keys (with fallbacks, so existing configs keep
working):

| Key | Default | Purpose |
|---|---|---|
| `client` | `"opencode"` | which agent `ronin opencode`/`claude` launches |
| `opencode_bin` | `"opencode"` | executable name/path on PATH |

`served_name` is reused as the opencode model id (already consistent with
vLLM's `--served-model-name`). No change to `Profile`.

### 2. `pod.py` — replace `claude_env()` with `opencode_config()`

`claude_env()` (pod.py:285) builds Anthropic env vars. Replace it with a
function that returns the **opencode config dict** (and keep a thin env helper
for the API key). Mapping:

| Claude Code (now) | opencode (new) |
|---|---|
| `ANTHROPIC_BASE_URL = <url>` | `provider.ronin.options.baseURL = <url>/v1` |
| `ANTHROPIC_AUTH_TOKEN = <key>` | `provider.ronin.options.apiKey = "{env:RONIN_API_KEY}"` |
| `ANTHROPIC_MODEL = served_name` | top-level `model = "ronin/<served_name>"` |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | `models.<served_name>.limit.context` (from `max_model_len()`) |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `models.<served_name>.limit.output` (16384) |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | *dropped* — opencode manages its own context window |
| attribution / betas / nonessential-traffic flags | *dropped* — Claude-Code-specific |

Generated config (written by ronin, see §4):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "ronin/<served_name>",
  "provider": {
    "ronin": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ronin (self-hosted)",
      "options": {
        "baseURL": "https://<pod>-8000.proxy.runpod.net/v1",
        "apiKey": "{env:RONIN_API_KEY}"
      },
      "models": {
        "<served_name>": {
          "name": "<served_name>",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": <max_model_len>, "output": 16384 }
        }
      }
    }
  }
}
```

Notes:
- `baseURL` = `proxy_url(pod_id, port)` + `"/v1"`. vLLM serves OpenAI there today.
- `capabilities.tools: true` is required — opencode cannot autodetect it for a
  custom provider, and agentic coding depends on it. vLLM's existing
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder` stays as-is.
- `apiKey` uses `{env:RONIN_API_KEY}` so the secret is never written to the file;
  ronin passes the key through the environment at launch.

### 3. `cli.py` — `cmd_claude` → `cmd_opencode`

Rework `cmd_claude` (cli.py:621):

1. Resolve the binary: `shutil.which(cfg.opencode_bin)`; error with the
   opencode install URL if missing (mirror the current message).
2. Same pod liveness checks and `--up` handling (unchanged logic).
3. Write the generated config to **`.ronin/opencode.json`** (next to state).
4. Launch: `subprocess.call([exe, *passthrough_args], env={**os.environ,
   "OPENCODE_CONFIG": "<.ronin/opencode.json>", "RONIN_API_KEY": state["api_key"]})`.
   Optionally add `--model ronin/<served_name>` (CLI flag wins over any project
   `opencode.json` the user already has).

Using `OPENCODE_CONFIG` (loaded between global and project config) keeps ronin's
provider out of the user's own project config, so it never clobbers their setup.

Keep the subcommand name `claude` as a **hidden alias** for one release so muscle
memory and scripts don't break; add `opencode` (and maybe `code`) as the primary
name. `split_claude_args` / the `UNPADDED` / `BILLING_HINT` tuples just need the
new verb added.

### 4. `cmd_env`

`ronin env` currently prints shell `export`s for Claude Code. Repurpose it to
print the path of the generated `.ronin/opencode.json` (and the `OPENCODE_CONFIG`
/ `RONIN_API_KEY` exports), so a user can launch opencode by hand.

### 5. `cmd_doctor`

Replace the `shutil.which("claude")` check (cli.py:151) with `cfg.opencode_bin`,
and the install hint URL.

### 6. Docs

- `README.md`: command table (`claude` → `opencode`), the "point the agent at the
  pod" section, and any `ronin claude` examples.
- `CLAUDE.md`: the architecture diagram line (`ronin claude ──► Claude Code`) and
  the command list in "Layout".

### opencode gotcha (verified 2026-10-21, opencode v2.1.2)

opencode runs a **shared background service** that does **not** pick up
`OPENCODE_CONFIG` from the environment — so a config passed that way is invisible
to `opencode run` and to the TUI, which both talk to that service (symptom:
`Error: Model unavailable: ronin/ronin`). ronin therefore launches opencode with
`--standalone` (a private in-process server that *does* read `OPENCODE_CONFIG`).
`--standalone` is a global flag but must sit **after** the subcommand when there
is one (`opencode run --standalone …`) and lead for the bare TUI
(`opencode --standalone`); `opencode_standalone()` in cli.py places it. Verified
end to end against a live pod: basic chat, file **read** and file **write** tool
calls all worked through vLLM's `qwen3_coder` parser.

### Phase 1 testing checklist

- [ ] `ronin up` (profile `main`), then `ronin opencode` opens and answers a prompt.
- [ ] **Tool calling works** (ask opencode to read/edit a file — verifies the
      `qwen3_coder` parser + `capabilities.tools`). This is the main risk.
- [ ] Context limit respected (long session compacts instead of erroring).
- [ ] `ronin opencode --up` cold-starts a pod then launches.
- [ ] `.ronin/opencode.json` contains no secret; key arrives via env.
- [ ] `ronin env` output launches opencode by hand.
- [ ] `ronin claude` alias still works.

---

## Phase 2 — non-vLLM backends (GGUF) — DONE (llama.cpp), 2026-10-21

Done for **llama.cpp**, triggered by an internal model-comparison audit (v3),
whose primary picks are mostly GGUF. SGLang (for the DeepSeek `huge-*` picks) is
still pending — profiles that need it ship commented out in `ronin.toml`.

As built:

1. `Profile` gained `backend` (`"vllm" | "llamacpp"`, default `"vllm"`),
   `llamacpp_args`, `hf_file`, `image`; `config.py` validates the backend and
   appends `extra_llamacpp_args`.
2. `pod_body` dispatches to `BACKENDS[backend]` (`_vllm_spec` / `_llamacpp_spec`),
   each returning `(env, entrypoint, cmd, image)`; the common body (gpu, disk,
   guard-wrap, mounts) is shared. llama.cpp: `llama-server -hf <model>[:quant]`
   (+ `-hff`), `--alias <served_name>`, `--metrics`, `--jinja`, `*llamacpp_args`;
   key via `LLAMA_API_KEY`, cache via `LLAMA_CACHE` on a volume; default image
   `ghcr.io/ggml-org/llama.cpp:server-cuda`.
3. The guard reads its Prometheus metric names from the env; ronin passes the
   llama.cpp names (`llamacpp:requests_processing` / `…requests_deferred` /
   `…n_decode_total`). `/health` is identical, so `cmd_up` is unchanged.
4. `serve_arg` / `max_model_len` read the backend's own args (`-c` vs
   `--max-model-len`). The memory check skips GGUF (no safetensors metadata).
5. opencode config is byte-for-byte the same — both backends expose OpenAI `/v1`.

Decisions taken: keep `vllm_args` and add a parallel `llamacpp_args` (explicit,
backward-compatible) rather than one generic list; the GGUF quant rides in `model`
(`repo:quant`, llama.cpp-native) with `hf_file` for odd filenames; `reasoning_effort`
is a client setting, so it lives in profile comments, not serve flags.

**Image gotchas (found on the first live boot, image build b11783):** the
`server-cuda` image keeps the binary at `/app/llama-server` with `/app` *not* on
`PATH` (so ronin uses the absolute path), and ships **no Python** (only
`libgomp1 curl ffmpeg`). The guard therefore runs under a small `sh` bootstrap
(`LLAMACPP_GUARD_SH`) that apt-installs `python3` on boot and then `exec`s the guard,
whose source travels in `RONIN_GUARD_SRC`. If the install fails the container exits
and `ronin up` deletes the pod — it never serves unguarded.

**Verified end to end (2026-10-21, `basic-general` on an RTX 4090, $0.89/h):** ready
in 1m 41s; guard active in the pod; `/v1` rejects requests without the key (401);
model id `ronin`; a raw tool call came back as structured `tool_calls` (45.6 tok/s,
reasoning separated into `reasoning_content`); `ronin opencode run` read a file,
wrote a file and answered correctly; `.ronin/opencode.json` holds no secret.

**Verified `basic-coding` with MTP (same day, RTX 4090):** the mradermacher
Q4_K_M carries the MTP head (`qwen35.nextn_predict_layers = 1`, `blk.64.nextn.*`),
and llama-server logs `creating MTP draft context`. Ready in 1m 50s; 73 % draft
acceptance (1026/1410, `--spec-draft-n-max 3`); 113.6 tok/s decoding a 1497-token
answer. opencode ran a full agentic loop (wrote `roman.py` + tests, ran them via its
shell tool, edited, re-ran: 8/8 OK).

**Verified `basic-science`, `medium-general` (vLLM, `qwen3_coder` parser),
`medium-coding` (static-repo Q8_0, MTP 60 %)** — `medium-science` shares
`medium-general`'s serve args.

**Guard on llama.cpp:** llama-server only leaves `/health` public, so `/metrics`
answered the guard's key-less polls with 401 and the idle limit never fired. The
guard now sends the pod's own key (and reports a 401/403 once; a 503 just means the
model is still loading). Verified live: a `--idle 2` pod deleted itself 156 s after
it was ready.

**Community hosts:** two 4090 community hosts in a row had a dead GPU
(`failed to initialize CUDA`), on which llama.cpp falls back to CPU and still looks
healthy. `ronin up` now aborts and deletes the pod on those log lines, read from a
fresh log snapshot every 20 s (the long-lived stream delivered them ~3 min late).

**Verified `big-general` (RTX PRO 6000, secure via `--secure` fallback, $2.49/h):**
SC117's two-shard IQ3_S downloaded with `-hf repo:IQ3_S` (both shards, from the
`IQ3_S/` subfolder) and served 196K context; ready in 6m 09s; tool call OK; 100.3 tok/s
without MTP (MoE, 6B active); opencode's roman task 6/6 OK. Real cost $0.32. The
`compress_ratios` caveat was a misread: SC117's header has the right values
(`[0,0,0,4,…]`); the zeros are in the alternative GGUFs. `big-coding` shares its serve
args; `big-science` only adds KV q8_0.

**Still to test:** `huge-general` (and `huge-coding/science`, which need SGLang).

---

## Risks & open decisions

- **Tool-calling quality is the real risk**, not the plumbing. opencode drives
  tools over OpenAI function-calling; verify the abliterated models behave well
  through it before trusting any `*-coding` profile.
- **`eval/` harness**: it scores the *model's* responses against the endpoint, so
  it stays valid. But "how well it codes" is now mediated by a different agent —
  note that when comparing to earlier Claude Code runs.
- **Decision — command naming**: `ronin opencode`, `ronin code`, or keep `claude`
  as the generic verb? (Recommend: `opencode` primary, `claude` hidden alias.)
- **Decision — config location**: `OPENCODE_CONFIG` → `.ronin/opencode.json`
  (recommended, non-invasive) vs. writing the user's project `opencode.json`.
- **Decision — keep Claude Code support at all?** `client = "claude" | "opencode"`
  in `[defaults]` lets both coexist; costs a branch in `cmd_*`. Worth it if you
  still sometimes point Claude Code at the pod.

## Rough effort

Phase 1 is a focused change to three files (`config.py`, `pod.py`, `cli.py`) plus
docs — small, since no pod/serving logic moves. Phase 2 is larger and separate.
