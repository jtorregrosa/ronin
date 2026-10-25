"""Tests for scout/run.py (no network: the Hugging Face calls are not exercised)."""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("scout_run", ROOT / "scout" / "run.py")
scout = importlib.util.module_from_spec(_spec)
sys.modules["scout_run"] = scout  # dataclasses look their module up while being built
_spec.loader.exec_module(scout)

GIB = 1024 ** 3


def watch(**overrides) -> "scout.Watch":
    base = dict(
        keywords=["abliterated"], window_days=21, max_candidates=40,
        trusted={"bartowski"}, ignored={"spammer"}, min_downloads=50, min_likes=3,
        skip_tags={"mlx", "exl3"}, min_params_b=14,
        families={"q27": [re.compile(r"qwen3\.8-27b", re.I)]},
        tiers={"basic": scout.Tier("basic", 24, 0.34, 5),
               "medium": scout.Tier("medium", 48, 0.79, 10)},
        judge_model="", judge_timeout=60,
    )
    base.update(overrides)
    return scout.Watch(**base)


def repo(id_, tags=("gguf",), downloads=0, likes=0, params=27e9, base=None):
    return {"id": id_, "tags": list(tags), "downloads": downloads, "likes": likes,
            "gguf": {"total": params, "architecture": "qwen35"},
            "cardData": {"base_model": base} if base else {},
            "createdAt": "2026-10-20T00:00:00.000Z"}


class WatchFile(unittest.TestCase):
    def test_committed_watch_toml_loads(self):
        w = scout.load_watch()
        self.assertIn("abliterated", w.keywords)
        self.assertEqual(w.tiers["huge"].gpu_count, 4)
        self.assertTrue(w.families)

    def test_families_cover_live_profiles(self):
        # every live profile's model should belong to a watched family, or scout cannot
        # map candidates onto it
        w = scout.load_watch()
        raw = tomllib.loads((ROOT / "ronin.toml").read_text(encoding="utf-8"))
        for name, table in raw["profiles"].items():
            if "model" in table:
                self.assertIsNotNone(scout.family_of([table["model"]], w.families), name)


class QuantNames(unittest.TestCase):
    def test_labels(self):
        cases = {
            "orcarouter_Qwen3.8-27B-Uncensored-Q4_K_M.gguf": "Q4_K_M",
            "x.i1-IQ3_XXS.gguf": "IQ3_XXS",
            "Q8_0/model-Q8_0-00001-of-00002.gguf": "Q8_0",
            "IQ3_S/model-00001-of-00002.gguf": "IQ3_S",
            "Mix-128G/DeepSeek-Mix-128G-00002-of-00005.gguf": "Mix-128G",
            "model-bf16.gguf": "BF16",
            "model-Q8_0-Q4_K_M.gguf": "Q4_K_M",
        }
        for path, label in cases.items():
            self.assertEqual(scout.quant_label(path), label, path)

    def test_shards_add_up_and_aux_files_are_skipped(self):
        files = [
            {"path": "README.md", "size": 10},
            {"path": "m-Q4_K_M.gguf", "size": 10, "lfs": {"size": 17 * GIB}},
            {"path": "Q8_0/m-Q8_0-00001-of-00002.gguf", "size": 20 * GIB},
            {"path": "Q8_0/m-Q8_0-00002-of-00002.gguf", "size": 8 * GIB},
            {"path": "mmproj-m-f16.gguf", "size": GIB},
            {"path": "mtp/mtp-m-Q8_0.gguf", "size": GIB},
            {"path": "dflash2/m-DFlash2-Q8_0.gguf", "size": GIB},
            {"path": "m-Aggressive-FastMTP-32K.gguf", "size": GIB},
            {"path": "imatrix.gguf", "size": GIB},
        ]
        self.assertEqual(scout.gguf_quants(files), {"Q4_K_M": 17 * GIB, "Q8_0": 28 * GIB})


class Filter(unittest.TestCase):
    def test_reasons(self):
        w = watch()
        self.assertIsNone(scout.skip_reason(repo("bartowski/Qwen3.8-27B-abliterated-GGUF"), w))
        self.assertEqual(scout.skip_reason(repo("spammer/x"), w), "ignored author")
        self.assertEqual(scout.skip_reason(repo("a/x", tags=("mlx", "safetensors")), w),
                         "format ronin cannot serve")
        self.assertEqual(scout.skip_reason(repo("a/x", tags=("transformers",)), w), "no weights")
        self.assertEqual(scout.skip_reason(repo("bartowski/x", params=4e9), w), "too small")
        self.assertEqual(scout.skip_reason(repo("a/x"), w), "no traction yet")
        self.assertIsNone(scout.skip_reason(repo("a/x", downloads=60), w))
        self.assertIsNone(scout.skip_reason(repo("a/x", likes=3), w))

    def test_vision_projector_size_is_not_the_model_size(self):
        r = repo("bartowski/x")
        r["gguf"] = {"total": 4e8, "architecture": "clip"}
        self.assertIsNone(scout.param_count(r))
        self.assertIsNone(scout.skip_reason(r, watch()))

    def test_untrusted_outside_families_dropped_trusted_first(self):
        skipped = {}
        repos = [repo("a/Qwen3.8-27B-x", downloads=500), repo("a/Other-70B", downloads=900),
                 repo("bartowski/Other-70B"), repo("bartowski/Qwen3.8-27B-y")]
        kept = scout.order_candidates(repos, watch(max_candidates=2), skipped)
        self.assertEqual([r["id"] for r in kept],
                         ["bartowski/Qwen3.8-27B-y", "bartowski/Other-70B"])
        self.assertEqual(skipped, {"untrusted, unwatched family": 1, "over max_candidates": 1})

    def test_family_from_base_model(self):
        r = repo("someone/Cold-Fusion-GGUF", base="Luni/Qwen3.8-27B-Cold-Fusion")
        self.assertEqual(scout.family_of([r["id"], *scout.base_models(r)],
                                         watch().families), "q27")


class Baseline(unittest.TestCase):
    def test_committed_baseline_files_exist(self):
        w = scout.load_watch()
        self.assertTrue(w.baseline)
        for f in w.baseline:
            self.assertTrue(f.is_file(), f)
        self.assertIn("basic-general", w.baseline_text())

    def test_flags_repo_or_abliterated_parent_not_official_base(self):
        text = "alt_low_drift: 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored; base Qwen/Qwen3.8-27B"
        kw = ["abliterated", "heretic"]
        new = candidate(id="someone/Fresh-GGUF", base_models=["Qwen/Qwen3.8-27B"])
        self.assertEqual(scout.in_baseline(new, text, kw), [])
        quant = candidate(id="mradermacher/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF",
                          base_models=["0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored"])
        self.assertEqual(scout.in_baseline(quant, text, kw),
                         ["0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored"])


class TierFit(unittest.TestCase):
    def test_biggest_quant_leaving_kv_room(self):
        basic = watch().tiers["basic"]
        sizes = {"Q4_K_M": int(16.55 * GIB), "Q5_K_M": int(19.3 * GIB), "Q8_0": int(26.6 * GIB)}
        fit = scout.best_fit(sizes, basic, "gguf")
        self.assertEqual(fit.quant, "Q4_K_M")
        self.assertGreaterEqual(fit.kv_gib, basic.min_kv_gib)

    def test_vllm_reserves_more_than_llamacpp(self):
        tier = watch().tiers["basic"]
        self.assertLess(tier.usable_gib("safetensors"), tier.usable_gib("gguf"))

    def test_nothing_fits(self):
        self.assertIsNone(scout.best_fit({None: 60 * GIB}, watch().tiers["medium"], "safetensors"))


def candidate(**overrides) -> "scout.Candidate":
    base = dict(id="bartowski/Qwen3.8-27B-New-GGUF", format="gguf", created="2026-10-20",
                downloads=100, likes=5, trusted=True, family="q27",
                base_models=["Qwen/Qwen3.8-27B"], params_b=27.0, license="apache-2.0",
                gated=False, sizes={"Q4_K_M": 17 * GIB},
                fits={"basic": scout.TierFit("Q4_K_M", 17.0, 5.8)})
    base.update(overrides)
    return scout.Candidate(**base)


LIVE = [
    scout.LiveProfile("basic-general", "bartowski/orcarouter_Qwen3.8-27B-GGUF:Q4_K_M",
                      "llamacpp", "basic", "q27", ["NVIDIA GeForce RTX 4090"]),
    scout.LiveProfile("medium-general", "id-2/Qwen3.8-27B-FP8", "vllm", "medium", "q27",
                      ["NVIDIA L40S"]),
    scout.LiveProfile("medium-coding", "m/Qwen3.8-27B-GGUF:Q6_K", "llamacpp", "medium", "q27",
                      ["NVIDIA L40S"]),
]


class Proposals(unittest.TestCase):
    def test_same_backend_extends(self):
        text = scout.snippet(candidate(), "basic", LIVE[0], "cand-basic-x")
        parsed = tomllib.loads(text)["profiles"]["cand-basic-x"]
        self.assertEqual(parsed, {"extends": "basic-general",
                                  "model": "bartowski/Qwen3.8-27B-New-GGUF:Q4_K_M"})

    def test_base_prefers_same_backend_in_tier(self):
        c = candidate(fits={"medium": scout.TierFit("Q6_K", 22.0, 24.0)})
        self.assertEqual(scout.pick_base(c, "medium", LIVE).name, "medium-coding")
        st = candidate(format="safetensors", fits={"medium": scout.TierFit(None, 30.0, 8.0)})
        self.assertEqual(scout.pick_base(st, "medium", LIVE).name, "medium-general")

    def test_other_backend_gets_a_full_profile(self):
        st = candidate(format="safetensors", sizes={"": 30 * GIB},
                       fits={"basic": scout.TierFit(None, 30.0, 8.0)})
        parsed = tomllib.loads(scout.snippet(st, "basic", LIVE[0], "c"))["profiles"]["c"]
        self.assertEqual(parsed["backend"], "vllm")
        self.assertEqual(parsed["model"], st.id)
        self.assertNotIn("extends", parsed)
        self.assertEqual(parsed["gpus"], LIVE[0].gpus)

    def test_unique_names(self):
        taken = set()
        self.assertEqual([scout.unique("a", taken) for _ in range(3)], ["a", "a-2", "a-3"])

    def test_report_snippets_parse_as_toml(self):
        w = watch()
        cands = [candidate(), candidate(id="mradermacher/Qwen3.8-27B-New-GGUF"),
                 candidate(id="bartowski/Brand-New-100B", family=None, fits={})]
        text = scout.render(cands, LIVE, w, datetime(2026, 10, 4, tzinfo=timezone.utc),
                            {"too small": 3}, "", "2026-10-25")
        blocks = re.findall(r"```toml\n(.*?)```", text, re.S)
        self.assertEqual(len(blocks), 2)
        names = [next(iter(tomllib.loads(b)["profiles"])) for b in blocks]
        self.assertEqual(len(set(names)), 2)
        self.assertIn("## Outside the watched families", text)
        self.assertIn("3 too small", text)
        self.assertIn("_Not run (`--no-judge`)._", text)


if __name__ == "__main__":
    unittest.main()
