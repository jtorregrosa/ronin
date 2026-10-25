"""Scout Hugging Face for new abliterated models and propose ronin profiles.

A satellite of ronin, like eval/: it reads ronin.toml but never changes it, and never
starts a pod. Each run:

  1. discover  newest repos matching the keywords in watch.toml (Hugging Face API);
  2. enrich    format, base model, GGUF quant sizes / safetensors weights, and the
               biggest quant that fits each tier (same memory model as ronin/fit.py);
  3. judge     `claude -p` with scout/PROMPT.md reads the model cards, looks for
               independent evidence and gives a verdict (skip with --no-judge);
  4. propose   scout/reports/<date>.md: candidates per family, TOML snippets to paste
               into ronin.toml, and the A/B commands for eval/ with their cost.

Usage:
    python scout/run.py                 # full run, report in scout/reports/
    python scout/run.py --no-judge      # deterministic part only (no Claude)
    python scout/run.py --dry-run       # do not remember the reported repos
    python scout/run.py --days 7 --all  # other window; include already-reported repos

No external dependencies: stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCOUT_DIR = Path(__file__).resolve().parent
ROOT = SCOUT_DIR.parent
WATCH_PATH = SCOUT_DIR / "watch.toml"
PROMPT_PATH = SCOUT_DIR / "PROMPT.md"
REPORTS_DIR = SCOUT_DIR / "reports"

sys.path.insert(0, str(ROOT))

from ronin import USER_AGENT, ui  # noqa: E402
from ronin.fit import (DEFAULT_UTILIZATION, DTYPE_BYTES, GIB, OVERHEAD_GIB,  # noqa: E402
                       VRAM_GIB_PER_GB)

HF = "https://huggingface.co"
EXPAND = ["createdAt", "downloads", "likes", "tags", "cardData", "gated", "gguf", "safetensors"]
# Model card text handed to the judge, per candidate.
CARD_CHARS = 6000
# Quant token in a GGUF filename: Q4_K_M, IQ3_XXS, Q8_0, BF16, MXFP4_MOE...
QUANT = re.compile(r"(?<![A-Za-z0-9])(I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?)"
                   r"(?![A-Za-z0-9])", re.IGNORECASE)
SHARD = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.IGNORECASE)
# GGUF files that are not the weights of a quant.
# Matched against the whole path: drafts often live in their own folder (dflash2/, mtp/).
AUX_GGUF = re.compile(r"(?i)(^|[/_.-])(mmproj|mtp|draft|dflash|eagle|imatrix)")
# llama-server's compute buffers and CUDA context on top of weights and KV.
LLAMACPP_OVERHEAD_GIB = 1.0
# A GGUF smaller than this fraction of the biggest quant is a draft, not a quant.
DRAFT_RATIO = 0.1
# Licenses that need no reading before use; anything else is flagged.
COMMON_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "llama3.1", "llama3.3", "gemma"}


# ------------------------------------------------------------------------- config

@dataclass
class Tier:
    name: str
    vram_gb: float
    price_hour: float
    min_kv_gib: float
    gpu_count: int = 1

    def usable_gib(self, fmt: str) -> float:
        """Weights + KV budget. vLLM as ronin/fit.py counts it (it reserves
        --gpu-memory-utilization and graphs); llama.cpp takes nearly the whole card."""
        vram = self.vram_gb * self.gpu_count * VRAM_GIB_PER_GB
        if fmt == "gguf":
            return vram - LLAMACPP_OVERHEAD_GIB
        return vram * DEFAULT_UTILIZATION - OVERHEAD_GIB


@dataclass
class Watch:
    keywords: list[str]
    window_days: int
    max_candidates: int
    trusted: set[str]
    ignored: set[str]
    min_downloads: int
    min_likes: int
    skip_tags: set[str]
    min_params_b: float
    families: dict[str, list[re.Pattern]]
    tiers: dict[str, Tier]
    judge_model: str
    judge_timeout: float
    baseline: list[Path] = field(default_factory=list)

    def baseline_text(self) -> str:
        return "\n".join(f.read_text(encoding="utf-8") for f in self.baseline if f.exists())


def load_watch(path: Path = WATCH_PATH) -> Watch:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    s, f, j = raw.get("search", {}), raw.get("filter", {}), raw.get("judge", {})
    return Watch(
        keywords=list(s.get("keywords", ["abliterated"])),
        window_days=int(s.get("window_days", 21)),
        max_candidates=int(s.get("max_candidates", 40)),
        trusted={a.lower() for a in f.get("trusted_authors", [])},
        ignored={a.lower() for a in f.get("ignored_authors", [])},
        min_downloads=int(f.get("min_downloads", 50)),
        min_likes=int(f.get("min_likes", 3)),
        skip_tags={t.lower() for t in f.get("skip_tags", [])},
        min_params_b=float(f.get("min_params_b", 0)),
        families={name: [re.compile(p, re.IGNORECASE) for p in fam.get("patterns", [])]
                  for name, fam in raw.get("families", {}).items()},
        tiers={name: Tier(name, float(t["vram_gb"]), float(t.get("price_hour", 0)),
                          float(t.get("min_kv_gib", 4)), int(t.get("gpu_count", 1)))
               for name, t in raw.get("tiers", {}).items()},
        judge_model=j.get("model", ""),
        judge_timeout=float(j.get("timeout_minutes", 20)) * 60,
        baseline=[path.parent / f for f in raw.get("baseline", {}).get("files", [])],
    )


def family_of(texts: list[str], families: dict[str, list[re.Pattern]]) -> str | None:
    for name, patterns in families.items():
        if any(p.search(t) for p in patterns for t in texts):
            return name
    return None


def tier_of(profile_name: str, tiers: dict[str, Tier]) -> str | None:
    prefix = profile_name.split("-", 1)[0]
    return prefix if prefix in tiers else None


# ----------------------------------------------------------------- Hugging Face API

def hf_get(url: str, token: str | None, timeout: float = 20) -> tuple[bytes, str]:
    """Body and the `next` page URL (from the Link header, empty when there is none)."""
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                timeout=timeout) as resp:
        link = resp.headers.get("Link", "")
        body = resp.read()
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return body, match.group(1) if match else ""


def search_url(keyword: str) -> str:
    query = [("search", keyword), ("sort", "createdAt"), ("direction", "-1"),
             ("limit", "100")] + [("expand[]", e) for e in EXPAND]
    return f"{HF}/api/models?{urllib.parse.urlencode(query)}"


def discover(watch: Watch, since: datetime, token: str | None) -> list[dict]:
    """Repos created after `since` matching any keyword, newest first, deduplicated."""
    found: dict[str, dict] = {}
    for keyword in watch.keywords:
        url = search_url(keyword)
        while url:
            body, url = hf_get(url, token)
            page = json.loads(body)
            for repo in page:
                if parse_time(repo.get("createdAt")) >= since:
                    found.setdefault(repo["id"], repo)
            # sorted by createdAt: once a page reaches past the window, stop paging
            if not page or parse_time(page[-1].get("createdAt")) < since:
                break
    return sorted(found.values(), key=lambda r: r.get("createdAt", ""), reverse=True)


def repo_files(repo_id: str, token: str | None) -> list[dict]:
    url = f"{HF}/api/models/{urllib.parse.quote(repo_id, safe='/')}/tree/main?recursive=true"
    files = []
    while url:
        body, url = hf_get(url, token)
        files += [f for f in json.loads(body) if f.get("type") == "file"]
    return files


def model_card(repo_id: str, token: str | None) -> str:
    url = f"{HF}/{urllib.parse.quote(repo_id, safe='/')}/raw/main/README.md"
    try:
        body, _ = hf_get(url, token)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return ""
    return body.decode("utf-8", errors="replace")[:CARD_CHARS]


def parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# -------------------------------------------------------------------- filter/enrich

def author(repo_id: str) -> str:
    return repo_id.split("/", 1)[0]


def base_models(repo: dict) -> list[str]:
    base = (repo.get("cardData") or {}).get("base_model") or []
    return [base] if isinstance(base, str) else list(base)


def param_count(repo: dict) -> int | None:
    for key in ("safetensors", "gguf"):
        total = (repo.get(key) or {}).get("total")
        # a GGUF repo that only ships the vision projector reports the projector's size
        if total and not (key == "gguf" and repo["gguf"].get("architecture") == "clip"):
            return int(total)
    return None


def skip_reason(repo: dict, watch: Watch) -> str | None:
    """Why a discovered repo is not a candidate (None: it is one)."""
    who = author(repo["id"]).lower()
    if who in watch.ignored:
        return "ignored author"
    tags = {t.lower() for t in repo.get("tags", [])}
    if tags & watch.skip_tags:
        return "format ronin cannot serve"
    if "gguf" not in tags and "safetensors" not in tags:
        return "no weights"
    params = param_count(repo)
    if params is not None and params < watch.min_params_b * 1e9:
        return "too small"
    if who not in watch.trusted and (repo.get("downloads", 0) < watch.min_downloads
                                     and repo.get("likes", 0) < watch.min_likes):
        return "no traction yet"
    return None


def quant_label(path: str) -> str:
    """The quant a GGUF file belongs to: the last quant token of its name (Q4_K_M...),
    else its folder (IQ3_S/x.gguf, Mix-128G/x.gguf), else its name without the shard."""
    folder, _, name = path.rpartition("/")
    name = SHARD.sub("", name)
    tokens = QUANT.findall(name)
    if tokens:
        return tokens[-1].upper()  # "Q8_0-Q4_K_M" style names keep the quant at the end
    if folder:
        folder = folder.rsplit("/", 1)[-1]
        token = QUANT.fullmatch(folder)
        return token.group(1).upper() if token else folder
    return name[:-len(".gguf")]


def gguf_quants(files: list[dict]) -> dict[str, int]:
    """Bytes per quant, adding up the shards of split GGUFs. Projectors, MTP drafts and
    imatrix files are left out."""
    quants: dict[str, int] = {}
    for f in files:
        path = f.get("path", "")
        if not path.lower().endswith(".gguf") or AUX_GGUF.search(path):
            continue
        size = (f.get("lfs") or {}).get("size") or f.get("size") or 0
        quant = quant_label(path)
        quants[quant] = quants.get(quant, 0) + int(size)
    # drafts and projectors with unusual names (FastMTP-32K.gguf): far smaller than any quant
    floor = max(quants.values(), default=0) * DRAFT_RATIO
    return {q: size for q, size in quants.items() if size >= floor}


def safetensors_bytes(repo: dict) -> int | None:
    params = (repo.get("safetensors") or {}).get("parameters")
    if not params:
        return None
    return int(sum(count * DTYPE_BYTES.get(dtype, 2) for dtype, count in params.items()))


@dataclass
class TierFit:
    quant: str | None      # None for safetensors (the checkpoint as it is)
    size_gib: float
    kv_gib: float          # room left for the KV cache


def best_fit(sizes: dict[str | None, int], tier: Tier, fmt: str) -> TierFit | None:
    """The biggest weights that still leave tier.min_kv_gib for the KV cache."""
    best = None
    for quant, size in sizes.items():
        gib = size / GIB
        kv = tier.usable_gib(fmt) - gib
        if kv >= tier.min_kv_gib and (best is None or gib > best.size_gib):
            best = TierFit(quant, round(gib, 2), round(kv, 1))
    return best


@dataclass
class Candidate:
    id: str
    format: str                      # "gguf" or "safetensors"
    created: str
    downloads: int
    likes: int
    trusted: bool
    family: str | None
    base_models: list[str]
    params_b: float | None
    license: str
    gated: bool
    sizes: dict[str, int] = field(default_factory=dict)   # quant -> bytes ("" = checkpoint)
    fits: dict[str, TierFit] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    card: str = ""

    @property
    def url(self) -> str:
        return f"{HF}/{self.id}"


def enrich(repo: dict, watch: Watch, token: str | None, fetch_card: bool = True,
           baseline: str = "") -> Candidate:
    tags = {t.lower() for t in repo.get("tags", [])}
    fmt = "gguf" if "gguf" in tags else "safetensors"
    bases = base_models(repo)
    params = param_count(repo)
    card_data = repo.get("cardData") or {}
    cand = Candidate(
        id=repo["id"], format=fmt, created=(repo.get("createdAt") or "")[:10],
        downloads=int(repo.get("downloads", 0)), likes=int(repo.get("likes", 0)),
        trusted=author(repo["id"]).lower() in watch.trusted,
        family=family_of([repo["id"], *bases], watch.families), base_models=bases,
        params_b=round(params / 1e9, 1) if params else None,
        license=str(card_data.get("license_name") or card_data.get("license") or "unknown"),
        gated=bool(repo.get("gated")),
    )
    files: list[dict] = []
    if fmt == "gguf":
        try:
            files = repo_files(repo["id"], token)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            cand.flags.append(f"file list unavailable ({e})")
        cand.sizes = gguf_quants(files)
    else:
        size = safetensors_bytes(repo)
        if size:
            cand.sizes = {"": size}
    for name, tier in watch.tiers.items():
        fit = best_fit({q or None: s for q, s in cand.sizes.items()}, tier, fmt)
        if fit:
            cand.fits[name] = fit
    known = in_baseline(cand, baseline, watch.keywords)
    if known:
        cand.flags.insert(0, f"in the baseline: {', '.join(known)}")
    cand.flags += risk_flags(cand, files)
    if fetch_card:
        cand.card = model_card(repo["id"], token)
    return cand


def in_baseline(cand: Candidate, baseline: str, keywords: list[str]) -> list[str]:
    """Repos of this candidate the baseline study already evaluated: the repo itself, or
    an abliterated parent (official bases are in the baseline anyway, so they don't count)."""
    text = baseline.lower()
    parents = [b for b in cand.base_models if any(k.lower() in b.lower() for k in keywords)]
    return [r for r in [cand.id, *parents] if r.lower() in text]


def risk_flags(cand: Candidate, files: list[dict]) -> list[str]:
    flags = []
    if cand.gated:
        flags.append("gated (needs HF_TOKEN with access)")
    if not cand.base_models:
        flags.append("no base_model in the card")
    if cand.license not in COMMON_LICENSES | {"unknown"}:
        flags.append(f"license: {cand.license}")
    if not cand.sizes:
        flags.append("no weight files yet (upload in progress?)" if cand.format == "gguf"
                     else "weight size unknown")
    elif not cand.fits:
        flags.append("fits no tier")
    if any(not QUANT.fullmatch(q) for q in cand.sizes if q):
        flags.append("non-standard quant names (may need hf_file)")
    names = [f.get("path", "").lower() for f in files]
    if any("mtp" in n for n in names):
        flags.append("ships an MTP draft (try --spec-type draft-mtp)")
    if any("mmproj" in n for n in names):
        flags.append("vision projector present (add --no-mmproj for text-only)")
    return flags


def order_candidates(repos: list[dict], watch: Watch, skipped: dict[str, int]) -> list[dict]:
    """Drop untrusted repos outside every family (noise: only trusted authors surface new
    bases), then keep max_candidates: trusted and watched first, then by downloads."""
    def watched(r):
        return family_of([r["id"], *base_models(r)], watch.families) is not None

    def trusted(r):
        return author(r["id"]).lower() in watch.trusted

    kept = []
    for repo in repos:
        if watched(repo) or trusted(repo):
            kept.append(repo)
        else:
            skipped["untrusted, unwatched family"] = skipped.get("untrusted, unwatched family", 0) + 1
    kept.sort(key=lambda r: (not trusted(r), not watched(r), -r.get("downloads", 0)))
    if len(kept) > watch.max_candidates:
        skipped["over max_candidates"] = len(kept) - watch.max_candidates
        kept = kept[:watch.max_candidates]
    return kept


# ------------------------------------------------------------------------- profiles

@dataclass
class LiveProfile:
    name: str
    model: str
    backend: str
    tier: str | None
    family: str | None
    gpus: list[str]


def live_profiles(watch: Watch) -> list[LiveProfile]:
    from ronin.config import load_config

    cfg = load_config()
    return [LiveProfile(p.name, p.model, p.backend, tier_of(p.name, watch.tiers),
                        family_of([p.model], watch.families), p.gpus)
            for p in cfg.profiles.values()]


def slug(repo_id: str, limit: int = 40) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", repo_id.split("/", 1)[-1].lower()).strip("-")
    return name[:limit].rstrip("-")


def pick_base(cand: Candidate, tier: str, live: list[LiveProfile]) -> LiveProfile | None:
    """The live profile a candidate extends in a tier: one with the same backend if the
    tier has it (it inherits the serve flags), else the tier's first."""
    backend = "llamacpp" if cand.format == "gguf" else "vllm"
    in_tier = [p for p in live if p.tier == tier]
    return next((p for p in in_tier if p.backend == backend), in_tier[0] if in_tier else None)


def snippet(cand: Candidate, tier: str, base: LiveProfile, name: str) -> str:
    """A [profiles.cand-*] table to paste into ronin.toml."""
    fit = cand.fits[tier]
    model = f"{cand.id}:{fit.quant}" if fit.quant else cand.id
    backend = "llamacpp" if cand.format == "gguf" else "vllm"
    lines = [f"# {cand.url}  ({cand.created}, {fit.size_gib} GiB, ~{fit.kv_gib} GiB left for KV)",
             f"[profiles.{name}]"]
    if backend == base.backend:
        lines += [f'extends = "{base.name}"', f'model = "{model}"']
        if backend == "vllm":
            lines.append("# inherits the parsers and chat template: check they match this model")
    else:
        disk = max(40, int(math.ceil(fit.size_gib * 1.5 / 10) * 10))
        lines += [f'backend = "{backend}"', f'model = "{model}"',
                  f"gpus = {json.dumps(base.gpus)}", f"disk_gb = {disk}",
                  "# defaults: review context length and parsers"]
        if backend == "llamacpp":
            lines.append('llamacpp_args = ["-c", "65536", "-ngl", "99", "-fa", "on", "--no-mmproj"]')
        else:
            lines.append('vllm_args = ["--max-model-len", "65536", "--enable-auto-tool-choice", '
                         '"--enable-prefix-caching"]')
    return "\n".join(lines)


def unique(name: str, taken: set[str]) -> str:
    """`name`, or name-2, name-3... if a profile in the report already has it."""
    candidate, n = name, 1
    while candidate in taken:
        n += 1
        candidate = f"{name}-{n}"
    taken.add(candidate)
    return candidate


def ab_commands(name: str, base: str, tier: Tier) -> str:
    return (f"ronin up -p {base} ; python eval/run.py --label {base} --effort medium ; ronin down\n"
            f"ronin up -p {name} ; python eval/run.py --label {name} --effort medium ; ronin down\n"
            f"python eval/run.py --compare eval/results/{base}-*.json eval/results/{name}-*.json\n"
            f"# ~1 h per round at ${tier.price_hour:.2f}/h -> about ${2 * tier.price_hour:.2f}")


# --------------------------------------------------------------------------- report

def gb(nbytes: int) -> str:
    return f"{nbytes / GIB:.1f}"


def fits_cell(cand: Candidate) -> str:
    cells = []
    for tier, fit in cand.fits.items():
        cells.append(f"{tier} {fit.quant or 'as is'} ({fit.size_gib:g})")
    return ", ".join(cells) or "—"


def render(cands: list[Candidate], profiles: list[LiveProfile], watch: Watch,
           since: datetime, skipped: dict[str, int], judgment: str, today: str) -> str:
    out = [f"# scout report — {today}", "",
           f"Repos created since {since:%Y-%m-%d} matching "
           f"{', '.join(f'`{k}`' for k in watch.keywords)}. "
           f"{len(cands)} candidate(s); skipped: "
           + (", ".join(f"{n} {why}" for why, n in sorted(skipped.items())) or "none") + ".",
           "",
           "Generated by `python scout/run.py`. Sizes in GiB; a tier lists the biggest quant "
           "that leaves its `min_kv_gib` free (watch.toml). Nothing here is measured: "
           "validate with `eval/` before changing `ronin.toml`.", ""]
    names: set[str] = set()
    by_family: dict[str | None, list[Candidate]] = {}
    for c in cands:
        by_family.setdefault(c.family, []).append(c)
    for family in [*watch.families, None]:
        group = by_family.get(family)
        if not group:
            continue
        live = [p for p in profiles if p.family == family] if family else []
        title = f"## {family}" if family else "## Outside the watched families"
        out += [title, ""]
        if family:
            out.append("Live profiles: " + (", ".join(f"`{p.name}` ({p.model})" for p in live)
                                            or "none (watched only)") + ".")
        else:
            out.append("New bases or families not in watch.toml: add a family there if one "
                       "is worth following.")
        out += ["", "| Repo | Format | Base | Created | ↓ / ♥ | Fits | Flags |",
                "|---|---|---|---|---|---|---|"]
        for c in group:
            out.append(f"| [{c.id}]({c.url}){' ✓' if c.trusted else ''} | {c.format} | "
                       f"{', '.join(c.base_models) or '—'} | {c.created} | "
                       f"{c.downloads} / {c.likes} | {fits_cell(c)} | {'; '.join(c.flags) or '—'} |")
        out.append("")
        snippets = []
        for c in group:
            for tier in c.fits:
                base = pick_base(c, tier, live)
                if base:
                    name = unique(f"cand-{tier}-{slug(c.id)}", names)
                    snippets.append((snippet(c, tier, base, name),
                                     ab_commands(name, base.name, watch.tiers[tier])))
        if snippets:
            out += ["<details><summary>Profiles to paste into ronin.toml and A/B</summary>", ""]
            for toml_text, commands in snippets:
                out += ["```toml", toml_text, "```", "```powershell", commands, "```", ""]
            out += ["</details>", ""]
    out += ["## Judgment", "", judgment.strip() or "_Not run (`--no-judge`)._", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------- judge

def judge(cands: list[Candidate], watch: Watch) -> str:
    """Ask `claude -p` for a verdict per candidate. Returns markdown, or a note when
    Claude Code is missing or fails (the deterministic report is kept either way)."""
    exe = shutil.which("claude")
    if not exe:
        return "_Not run: `claude` is not on PATH._"
    config_text = (ROOT / "ronin.toml").read_text(encoding="utf-8")
    payload = [asdict(c) for c in cands]
    sections = [
        PROMPT_PATH.read_text(encoding="utf-8"),
        "## Current ronin.toml (the matrix and its evidence)",
        f"```toml\n{config_text}\n```",
    ]
    for f in watch.baseline:
        if f.exists():
            sections += [f"## Baseline study: {f.name}", f.read_text(encoding="utf-8")]
    prompt = "\n\n".join([
        *sections,
        "## Candidates (JSON; `card` is the start of each model card)",
        f"```json\n{json.dumps(payload, indent=1, ensure_ascii=False)}\n```",
    ])
    cmd = [exe, "-p", "--output-format", "text", "--allowedTools", "WebSearch", "WebFetch"]
    if watch.judge_model:
        cmd += ["--model", watch.judge_model]
    try:
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                encoding="utf-8", timeout=watch.judge_timeout, cwd=SCOUT_DIR)
    except subprocess.TimeoutExpired:
        return f"_Not run: `claude -p` took longer than {watch.judge_timeout / 60:g} min._"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["no output"]
        return f"_Not run: `claude -p` exited with {result.returncode} ({detail[0]})._"
    return result.stdout


# ---------------------------------------------------------------------------- state

def state_path() -> Path:
    from ronin.config import project

    return project.state_dir / "scout-state.json"


def load_seen() -> dict[str, str]:
    path = state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("reported", {})
    except (json.JSONDecodeError, OSError):
        return {}


def save_seen(seen: dict[str, str]) -> None:
    path = state_path()
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"reported": seen}, indent=1), encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scout Hugging Face for abliterated models "
                                                 "and propose ronin profiles.")
    parser.add_argument("--days", type=int, help="look back this many days (default: watch.toml)")
    parser.add_argument("--all", action="store_true", help="include already-reported repos")
    parser.add_argument("--no-judge", action="store_true", help="skip the `claude -p` verdict")
    parser.add_argument("--dry-run", action="store_true",
                        help="do not remember the reported repos (the report is still written)")
    args = parser.parse_args(argv)

    from ronin.config import ConfigError, load_env

    load_env()
    token = os.environ.get("HF_TOKEN") or None
    watch = load_watch()
    try:
        profiles = live_profiles(watch)
    except ConfigError as e:
        ui.error(str(e))
        return 1
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days or watch.window_days)
    seen = {} if args.all else load_seen()

    progress = ui.Progress("Searching Hugging Face")
    try:
        repos = discover(watch, since, token)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        progress.stop()
        ui.error(f"Could not reach Hugging Face\n{e}")
        return 1
    progress.stop()
    skipped: dict[str, int] = {}
    picked = []
    for repo in repos:
        why = "already reported" if repo["id"] in seen else skip_reason(repo, watch)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
        else:
            picked.append(repo)
    cands = order_candidates(picked, watch, skipped)
    ui.ok(f"{len(repos)} repos since {since:%Y-%m-%d}, {len(cands)} candidate(s)")

    baseline = watch.baseline_text()
    enriched = []
    progress = ui.Progress("Reading repos")
    for i, repo in enumerate(cands, 1):
        if ui.TTY:  # redirected, one line per repo would only be noise
            progress.phase(f"Reading repos ({i}/{len(cands)})")
            progress.tick()
        enriched.append(enrich(repo, watch, token, fetch_card=not args.no_judge,
                               baseline=baseline))
    progress.stop()

    judgment = ""
    if enriched and not args.no_judge:
        ui.step(f"Asking `claude -p` for a verdict on {len(enriched)} candidate(s)"
                f"{ui.SYM['dots']} (may take several minutes)")
        judgment = judge(enriched, watch)
    elif not enriched:
        judgment = "_No new candidates._"

    today = now.strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(exist_ok=True)
    report = REPORTS_DIR / f"{today}.md"
    report.write_text(render(enriched, profiles, watch, since, skipped, judgment, today),
                      encoding="utf-8")
    ui.ok(f"Report written: {report.relative_to(ROOT)}")
    if judgment.startswith("_Not run"):
        # keep them unreported so the next run judges them
        ui.warn(judgment.strip("_"))
        ui.hint("The candidates are not marked as reported; the next run retries them.")
    elif not args.dry_run:
        save_seen({**load_seen(), **{c.id: today for c in enriched}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
