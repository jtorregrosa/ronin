# scout

Finds new abliterated / uncensored uploads on Hugging Face, checks which of ronin's tiers
each one fits, asks Claude for a verdict on the evidence, and writes a report with
ready-to-paste profiles and the `eval/` A/B commands to validate them.

Like `eval/`, it is a satellite: it reads `ronin.toml` but **never changes it** and
**never starts a pod**. The paid step (the A/B) stays yours.

```
discover ──► enrich ──► judge ──► propose ──► (you) ronin up + eval/run.py
 HF API      fit        claude -p  reports/     validation, costs GPU time
 free        free       optional   <date>.md
```

## Usage

```powershell
python scout/run.py                 # full run; report in scout/reports/<date>.md
python scout/run.py --no-judge      # deterministic part only (no Claude)
python scout/run.py --dry-run       # do not remember the reported repos
python scout/run.py --days 7 --all  # other window; include already-reported repos
```

`HF_TOKEN` (from `.env`) is used when set: fewer rate limits and access to gated repos.
The judge needs [Claude Code](https://claude.com/claude-code) on `PATH`; it runs
`claude -p` with only `WebSearch` and `WebFetch` allowed, and takes a few minutes.

## What each phase does

1. **discover**: repos created in the last `window_days` whose id matches a keyword,
   newest first. Repos already in a report (`.ronin/scout-state.json`) are skipped, so
   each run only shows what is new. One that lacked traction comes back on a later run.
2. **filter**: drops formats ronin cannot serve (MLX, EXL3...), models under
   `min_params_b`, and untrusted uploads without downloads or likes. Untrusted repos
   outside every watched family are dropped too; trusted authors outside them are kept
   under *Outside the watched families*, so new base models still surface.
3. **enrich**: format, base model, license, the size of every GGUF quant (adding up
   shards; projectors and MTP drafts left out) or the safetensors weights. Per tier, the
   biggest quant that leaves `min_kv_gib` for the KV cache. vLLM is counted the same way
   as `ronin/fit.py`; llama.cpp uses nearly the whole card.
4. **judge**: `PROMPT.md` + the current `ronin.toml` + the candidates (with the start of
   each card) go to `claude -p`. It returns an A/B queue, a verdict per candidate with
   tagged evidence (`[IND]` / `[AUT]` / `[NV]`, as in `ronin.toml`), and notes on the
   matrix. If it fails, the report is still written and the candidates are retried on
   the next run.
5. **propose**: `reports/<date>.md`, one section per family. For every tier a candidate
   fits, it gives a `[profiles.cand-*]` table that extends the live profile of that tier
   with the same backend (or a full skeleton when the backend differs), plus the A/B
   commands and their approximate cost.

## Baseline

`baseline/` holds the study the live matrix was built from (copied from the original
comparison of 2026-10-21): `profiles-v3.yaml` (per profile: the live pick's `[IND]`
figures, the alternatives weighed, the discards and why) and the full report. It is
used twice:

- the judge gets it as the bar to beat, and is told not to re-propose alternatives or
  discards without newer evidence;
- the deterministic part flags `in the baseline` on any candidate whose repo, or whose
  abliterated parent (e.g. the repo a GGUF was quantized from), the study names.

When a new study or a won A/B supersedes it, replace the files (or add them to
`baseline.files` in `watch.toml`).

## Acting on a report

1. Read the *A/B queue* in the judgment and pick one.
2. Paste its `[profiles.cand-*]` table into `ronin.toml` (review the flags: `--no-mmproj`,
   MTP, parsers).
3. Run the A/B commands from the report (both rounds with the same `--effort`).
4. If it wins, promote it: replace the live profile's `model`, update `profiles.md`, and
   cite the report in the profile's comment.

## Configuration (`watch.toml`)

| Key | What it does |
|---|---|
| `search.keywords` | substrings searched in repo ids |
| `search.window_days` | how far back each run looks |
| `search.max_candidates` | cap on repos enriched per run |
| `filter.trusted_authors` / `ignored_authors` | always in / always out |
| `filter.min_downloads` / `min_likes` | traction needed by everyone else |
| `filter.skip_tags`, `min_params_b` | formats and sizes to ignore |
| `families.<name>.patterns` | regexes matched against the repo id, the base model and each profile's `model` |
| `tiers.<name>` | VRAM, GPU count, $/h and `min_kv_gib`; `<name>` is the profile prefix (`basic-general` → `basic`) |
| `baseline.files` | the study the judge measures candidates against (see *Baseline*) |
| `judge.model`, `judge.timeout_minutes` | `claude -p --model` and its time limit |

When you add a profile for a new base model, add its family here too (a test checks that
every live profile belongs to one).

## Running it on a schedule

Windows Task Scheduler, weekly (adjust the paths):

```powershell
$action = New-ScheduledTaskAction -Execute "python" -Argument "scout\run.py" `
  -WorkingDirectory "C:\path\to\abliterated-workspace"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 9am
Register-ScheduledTask -TaskName "ronin-scout" -Action $action -Trigger $trigger
```

Reports are meant to be committed: `profiles.md` and the profile comments can cite the
report a change came from.
