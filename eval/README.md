# eval

A mini-benchmark to compare model variants on the **same** tasks and pick the one that
works best for you. It measures what model cards don't: real throughput on a given GPU
and how the variants differ on your own prompts.

## What it measures

Per task in `tasks.jsonl`:
- **latency** and **tokens/s** (real throughput on that GPU);
- content **checks** (`expect_contains`): a heuristic for "did the answer go the right way?";
- **refusal** (`refused`): whether the model declines (refusal markers near the start).

Tasks carry a free-form `category` so you can group and compare however you like. The
summary and `--compare` report totals per label and per category.

## Tasks file

Your task set lives in `eval/tasks.jsonl`, which is **gitignored** — it holds your own
prompts and stays local. A small neutral `tasks.example.jsonl` is committed as a
starting point; `run.py` uses it automatically when `tasks.jsonl` is absent. To build
your own set, copy it:

```powershell
copy eval\tasks.example.jsonl eval\tasks.jsonl
```

## Usage

With a pod up (`ronin up -p <profile>`):

```powershell
python eval/run.py --label my-run       # run against the active pod
```

While a task streams, its line is rewritten in place with the elapsed seconds and
whether the model is thinking or writing; the final line keeps the usual format.

Recommended comparison flow (use the same `--effort` in every round):

```powershell
ronin up -p <a> ; python eval/run.py --label a --effort medium ; ronin down
ronin up -p <b> ; python eval/run.py --label b --effort medium ; ronin down

python eval/run.py --compare eval/results/a-*.json eval/results/b-*.json
```

If a task in a round fails (e.g. a network timeout), you don't need to rerun the whole
round. With the **same pod** from that round still up:

```powershell
python eval/run.py --retry-errors eval/results/a-XXXXXXXX-XXXXXX.json
```

It reruns only the errored tasks (with that round's effort) and re-scores the other
saved responses against the current checks in `tasks.jsonl`. Existing responses are not
regenerated. It refuses to run if the active pod is a different profile.

Against a manual endpoint (without ronin):

```powershell
python eval/run.py --base-url https://host:8000 --model ronin --api-key KEY --label test
```

## Reading the results

- **checks / tok/s across labels**: how the variants differ in answer quality and speed.
- **refusals**: `refused` counts per label, a keyword proxy — **review the responses by
  hand** in `results/*.json` before concluding.

## Extending the tasks

Add lines to `tasks.jsonl`. Schema:

```json
{"id": "my-task", "category": "coding", "prompt": "...", "expect_contains": ["x", "y|z"], "max_tokens": 1024}
```

`expect_contains` terms may list alternatives with `|`.
