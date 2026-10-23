"""Unit tests for ronin (standard library only): `python -m unittest`."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ronin import cli, config, fit, guard, pod, runpod, watchdog

pod_mod = pod   # `pod` is also a common local name in tests
from ronin.config import ConfigError
from ronin.runpod import RunPod, RunPodError

MINIMAL = """
[defaults]
profile = "main"

[profiles.main]
model = "org/model"
gpus = ["GPU A", "GPU B"]
vllm_args = ["--max-model-len", "65536", "--enable-prefix-caching"]
"""


class ProjectTestCase(unittest.TestCase):
    """Runs each test inside a throwaway project directory."""

    toml = MINIMAL

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "ronin.toml").write_text(self.toml, encoding="utf-8")
        self.previous = config.project
        config.use_project(str(self.root / "ronin.toml"))

    def tearDown(self):
        config.project = self.previous
        self.tmp.cleanup()


class SplitAgentArgs(unittest.TestCase):
    def check(self, argv, ronin, agent):
        self.assertEqual(cli.split_agent_args(argv), (ronin, agent))

    def test_unknown_options_pass_through(self):
        self.check(["opencode", "--resume"], ["opencode"], ["--resume"])
        self.check(["claude", "--resume"], ["claude"], ["--resume"])

    def test_print_mode_is_not_taken_as_profile(self):
        self.check(["opencode", "-p", "explain"], ["opencode"], ["-p", "explain"])

    def test_ronin_options_lead(self):
        self.check(["opencode", "--up", "--profile", "big", "--resume"],
                   ["opencode", "--up", "--profile", "big"], ["--resume"])
        self.check(["claude", "--profile=big", "x"], ["claude", "--profile=big"], ["x"])

    def test_stock_options_are_ronins(self):
        self.check(["opencode", "--up", "--secure", "--max-wait", "10", "--continue"],
                   ["opencode", "--up", "--secure", "--max-wait", "10"], ["--continue"])
        self.check(["opencode", "--max-wait=5", "x"], ["opencode", "--max-wait=5"], ["x"])

    def test_double_dash_is_dropped(self):
        self.check(["opencode", "--up", "--", "--up"], ["opencode", "--up"], ["--up"])

    def test_global_config_before_command(self):
        self.check(["--config", "a.toml", "opencode", "-c"],
                   ["--config", "a.toml", "opencode"], ["-c"])

    def test_other_commands_untouched(self):
        self.check(["up", "-p", "big"], ["up", "-p", "big"], [])


class OpencodeStandalone(unittest.TestCase):
    def test_bare_tui_leads_with_flag(self):
        self.assertEqual(cli.opencode_standalone([]), ["--standalone"])

    def test_flag_before_passed_options(self):
        self.assertEqual(cli.opencode_standalone(["--continue"]), ["--standalone", "--continue"])

    def test_flag_after_a_subcommand(self):
        self.assertEqual(cli.opencode_standalone(["run", "msg"]),
                         ["run", "--standalone", "msg"])


class Tokens(unittest.TestCase):
    def test_parse_tokens(self):
        self.assertEqual(pod.parse_tokens("131072"), 131072)
        self.assertEqual(pod.parse_tokens("128K"), 131072)
        self.assertEqual(pod.parse_tokens("100k"), 100_000)

    def test_max_model_len_forms(self):
        p = config.Profile("x", "m", ["g"], ["--max-model-len=32K"])
        self.assertEqual(pod.max_model_len(p), 32768)
        p.vllm_args = ["--max-model-len", "4096"]
        self.assertEqual(pod.max_model_len(p), 4096)
        p.vllm_args = []
        self.assertEqual(pod.max_model_len(p), pod.DEFAULT_CONTEXT)


class RootCause(unittest.TestCase):
    def test_prefers_specific_error_over_wrapper(self):
        lines = ["INFO starting",
                 "ValueError: max_model_len 262144 is larger than the model allows",
                 "RuntimeError: Engine core initialization failed. See root cause above."]
        self.assertTrue(pod.root_cause(lines).startswith("ValueError"))

    def test_none_without_errors(self):
        self.assertIsNone(pod.root_cause(["all fine"]))


class Profiles(ProjectTestCase):
    toml = MINIMAL + """
[profiles.small]
extends = "main"
model = "org/small"
extra_vllm_args = ["--max-num-seqs", "4"]
"""

    def test_extends_inherits_and_appends(self):
        cfg = config.load_config()
        small = cfg.profiles["small"]
        self.assertEqual(small.model, "org/small")
        self.assertEqual(small.gpus, ["GPU A", "GPU B"])
        self.assertEqual(small.vllm_args[-2:], ["--max-num-seqs", "4"])
        self.assertIn("--enable-prefix-caching", small.vllm_args)

    def test_guard_defaults(self):
        cfg = config.load_config()
        self.assertEqual((cfg.max_hours, cfg.idle_minutes), (8, 60))

    def test_extends_loop(self):
        raw = {"a": {"extends": "b"}, "b": {"extends": "a"}}
        with self.assertRaises(ConfigError):
            config.resolve_profile("a", raw)

    def test_unknown_default_profile(self):
        (self.root / "ronin.toml").write_text(MINIMAL.replace('profile = "main"', 'profile = "nope"'),
                                              encoding="utf-8")
        with self.assertRaises(ConfigError):
            config.load_config()

    def test_unknown_backend_rejected(self):
        (self.root / "ronin.toml").write_text(
            MINIMAL + '\n[profiles.x]\nbackend = "sglang"\nmodel = "o/m"\ngpus = ["G"]\n',
            encoding="utf-8")
        with self.assertRaises(ConfigError):
            config.load_config()

    def test_bad_min_cuda_rejected(self):
        (self.root / "ronin.toml").write_text(MINIMAL + 'min_cuda = "12"\n', encoding="utf-8")
        with self.assertRaises(ConfigError):
            config.load_config()

    def test_llamacpp_args_inherit_and_append(self):
        (self.root / "ronin.toml").write_text(MINIMAL + """
[profiles.gg]
backend = "llamacpp"
model = "o/m-GGUF:Q4_K_M"
gpus = ["G"]
llamacpp_args = ["-c", "65536"]

[profiles.gg2]
extends = "gg"
extra_llamacpp_args = ["-ngl", "99"]
""", encoding="utf-8")
        cfg = config.load_config()
        self.assertEqual(cfg.profiles["gg2"].backend, "llamacpp")
        self.assertEqual(cfg.profiles["gg2"].llamacpp_args, ["-c", "65536", "-ngl", "99"])


class FindConfig(unittest.TestCase):
    def test_searches_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ronin.toml").write_text(MINIMAL, encoding="utf-8")
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"RONIN_CONFIG": ""}):
                self.assertEqual(config.find_config(start=nested).resolve(),
                                 (root / "ronin.toml").resolve())

    def test_explicit_and_env_win(self):
        self.assertEqual(config.find_config("x.toml"), Path("x.toml"))
        with mock.patch.dict(os.environ, {"RONIN_CONFIG": "y.toml"}):
            self.assertEqual(config.find_config(), Path("y.toml"))


class PodBody(ProjectTestCase):
    def test_key_in_env_not_command_line(self):
        cfg = config.load_config()
        body = pod.pod_body(cfg, cfg.profile(None), "GPU A", None, "secret")
        self.assertEqual(body["env"]["VLLM_API_KEY"], "secret")
        self.assertNotIn("secret", body["cmd"])
        self.assertNotIn("--api-key", body["cmd"])
        self.assertEqual(body["entrypoint"], ["vllm", "serve"])

    def test_guard_wraps_vllm(self):
        cfg = config.load_config()
        body = pod.pod_body(cfg, cfg.profile(None), "GPU A", None, "k",
                            deadline=1_900_000_000, idle_seconds=3600)
        self.assertEqual(body["entrypoint"][:2], ["python3", "-c"])
        self.assertEqual(body["entrypoint"][-2:], ["vllm", "serve"])
        self.assertEqual(body["env"]["RONIN_DEADLINE"], "1900000000")
        self.assertEqual(body["env"]["RONIN_IDLE"], "3600")

    def test_vllm_has_no_cuda_filter_unless_set(self):
        cfg = config.load_config()
        p = cfg.profile(None)
        self.assertNotIn("minCudaVersion", pod.pod_body(cfg, p, "GPU A", None, "k")["gpu"])
        p.min_cuda = "12.9"
        self.assertEqual(pod.pod_body(cfg, p, "GPU A", None, "k")["gpu"]["minCudaVersion"], "12.9")

    def test_single_gpu_has_no_tensor_parallel(self):
        cfg = config.load_config()
        body = pod.pod_body(cfg, cfg.profile("main"), "GPU A", None, "k")
        self.assertEqual(body["gpu"]["count"], 1)
        self.assertNotIn("--tensor-parallel-size", body["cmd"])

    def test_multi_gpu_adds_tensor_parallel(self):
        cfg = config.load_config()
        p = config.Profile("tp", "org/m", ["GPU A"], [], gpu_count=4)
        body = pod.pod_body(cfg, p, "GPU A", None, "k")
        self.assertEqual(body["gpu"]["count"], 4)
        i = body["cmd"].index("--tensor-parallel-size")
        self.assertEqual(body["cmd"][i + 1], "4")

    def test_ephemeral_has_no_vllm_cache_root(self):
        cfg = config.load_config()
        body = pod.pod_body(cfg, cfg.profile("main"), "GPU A", None, "k")
        self.assertNotIn("VLLM_CACHE_ROOT", body["env"])

    def test_volume_persists_vllm_compile_cache(self):
        cfg = config.load_config()
        body = pod.pod_body(cfg, cfg.profile("main"), "GPU A", "vol-1", "k")
        self.assertEqual(body["env"]["VLLM_CACHE_ROOT"], f"{pod.VOLUME_MOUNT}/vllm")
        self.assertTrue(body["env"]["HF_HOME"].startswith(pod.VOLUME_MOUNT))

    def test_chat_template_relative_to_config(self):
        (self.root / "t.jinja").write_text("{{ x }}", encoding="utf-8")
        cfg = config.load_config()
        profile = cfg.profile(None)
        profile.chat_template = "t.jinja"
        body = pod.pod_body(cfg, profile, "GPU A", None, "k")
        self.assertEqual(body["cmd"][-2:], ["--chat-template", "{{ x }}"])


class LlamacppPodBody(ProjectTestCase):
    toml = """
[defaults]
profile = "gg"
image = "vllm/vllm-openai:v0.32.0"

[profiles.gg]
backend = "llamacpp"
model = "org/repo-GGUF:Q4_K_M"
gpus = ["GPU A"]
hf_file = "repo-Q4_K_M-mtp.gguf"
llamacpp_args = ["-c", "65536", "-ngl", "99", "-ctk", "q8_0"]
"""

    def body(self, **kw):
        cfg = config.load_config()
        return pod.pod_body(cfg, cfg.profile(None), "GPU A", kw.pop("volume", None), "secret", **kw)

    def test_llama_server_command(self):
        body = self.body()
        # absolute path: the image's PATH does not include /app
        self.assertEqual(body["entrypoint"], [pod.LLAMACPP_BIN])
        self.assertEqual(body["cmd"][:2], ["-hf", "org/repo-GGUF:Q4_K_M"])
        self.assertIn("--jinja", body["cmd"])
        self.assertIn("--metrics", body["cmd"])
        # the served name is exposed via --alias (what opencode addresses)
        self.assertEqual(body["cmd"][body["cmd"].index("--alias") + 1], "ronin")
        # the exact GGUF file is passed with -hff
        self.assertEqual(body["cmd"][body["cmd"].index("-hff") + 1], "repo-Q4_K_M-mtp.gguf")
        # the profile's llama args are appended verbatim
        self.assertEqual(body["cmd"][-6:], ["-c", "65536", "-ngl", "99", "-ctk", "q8_0"])

    def test_key_in_env_not_command_line(self):
        body = self.body()
        self.assertEqual(body["env"]["LLAMA_API_KEY"], "secret")
        self.assertNotIn("VLLM_API_KEY", body["env"])
        self.assertNotIn("secret", body["cmd"])
        self.assertNotIn("--api-key", body["cmd"])

    def test_default_image_is_llamacpp_not_vllm(self):
        self.assertEqual(self.body()["image"], pod.LLAMACPP_IMAGE)

    def test_volume_persists_gguf_cache(self):
        body = self.body(volume="vol-1")
        self.assertEqual(body["env"]["LLAMA_CACHE"], f"{pod.VOLUME_MOUNT}/llama")

    def test_guard_injects_llamacpp_metrics(self):
        body = self.body(deadline=1_900_000_000, idle_seconds=3600)
        # the image has no Python: a sh bootstrap installs it, then runs the guard
        self.assertEqual(body["entrypoint"][:2], ["sh", "-c"])
        self.assertIn("exec python3", body["entrypoint"][2])
        self.assertEqual(body["entrypoint"][-1], pod.LLAMACPP_BIN)
        self.assertIn("def main", body["env"]["RONIN_GUARD_SRC"])
        self.assertIn("llamacpp:requests_processing", body["env"]["RONIN_BUSY_METRICS"])
        self.assertEqual(body["env"]["RONIN_DONE_METRIC"], "llamacpp:n_decode_total")

    def test_llamacpp_pods_require_cuda_12_8_hosts(self):
        self.assertEqual(self.body()["gpu"]["minCudaVersion"], "12.8")

    def test_context_len_reads_ctx_size(self):
        self.assertEqual(pod.max_model_len(config.load_config().profile(None)), 65536)


class Retries(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(runpod.time, "sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_get_retries_transient_errors(self):
        rp = RunPod("k")
        with mock.patch.object(rp, "_send", side_effect=[RunPodError(0, "down"),
                                                          RunPodError(502, "bad gateway"),
                                                          {"id": "p"}]) as send:
            self.assertEqual(rp.get_pod("p"), {"id": "p"})
        self.assertEqual(send.call_count, 3)

    def test_post_is_never_retried(self):
        rp = RunPod("k")
        with mock.patch.object(rp, "_send", side_effect=RunPodError(503, "busy")) as send:
            with self.assertRaises(RunPodError):
                rp.create_pod({})
        self.assertEqual(send.call_count, 1)

    def test_client_errors_are_not_retried(self):
        rp = RunPod("k")
        with mock.patch.object(rp, "_send", side_effect=RunPodError(404, "nope")) as send:
            with self.assertRaises(RunPodError):
                rp.get_pod("p")
        self.assertEqual(send.call_count, 1)


class FakeRunPod:
    def __init__(self, pod=None, error=None):
        self.pod, self.error, self.deleted = pod, error, []

    def get_pod(self, pod_id, timeout=None):
        if self.error:
            raise self.error
        return self.pod

    def delete_pod(self, pod_id):
        self.deleted.append(pod_id)

    def account(self):
        return {"balance": 10.0, "spend_per_hour": 0.0}


class WatchdogCheck(ProjectTestCase):
    def setUp(self):
        super().setUp()
        self.state = {"pod_id": "p1", "api_key": "k", "volume_id": "v"}
        config.save_state(self.state)
        patcher = mock.patch.object(watchdog.mascot, "show")
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, rp, limit=3600):
        with mock.patch("sys.stdout"):
            return watchdog._wd_check(rp, "p1", limit, 0)

    def test_under_limit_returns_pod(self):
        rp = FakeRunPod({"status": "RUNNING", "runtime": {"uptime": 10}})
        self.assertIsNotNone(self.check(rp))
        self.assertEqual(rp.deleted, [])

    def test_over_limit_deletes(self):
        rp = FakeRunPod({"status": "RUNNING", "runtime": {"uptime": 4000}, "cost": 1.0})
        self.assertIsNone(self.check(rp))
        self.assertEqual(rp.deleted, ["p1"])
        self.assertNotIn("pod_id", config.load_state())
        self.assertEqual(config.load_state()["volume_id"], "v")   # the volume is kept

    def test_dead_pod_is_deleted(self):
        rp = FakeRunPod({"status": "EXITED"})
        self.assertIsNone(self.check(rp))
        self.assertEqual(rp.deleted, ["p1"])

    def test_gone_pod_clears_state(self):
        self.assertIsNone(self.check(FakeRunPod(error=RunPodError(404, "gone"))))
        self.assertNotIn("pod_id", config.load_state())

    def test_transient_error_propagates_for_retry(self):
        with self.assertRaises(RunPodError) as ctx:
            self.check(FakeRunPod(error=RunPodError(0, "offline")))
        self.assertTrue(ctx.exception.transient)


class StaleState(ProjectTestCase):
    """A long-running `up` / `watchdog` must not unregister a pod created after `ronin down`."""

    def setUp(self):
        super().setUp()
        # p1 was deleted with `ronin down`; a new `up` then registered p2
        config.save_state({"pod_id": "p2", "api_key": "k2", "volume_id": "v",
                           "history": [{"pod_id": "p1"}, {"pod_id": "p2"}]})

    def test_terminate_own_leaves_newer_pod_registered(self):
        rp = FakeRunPod()
        pod.terminate_own(rp, "p1")
        self.assertEqual(rp.deleted, ["p1"])
        state = config.load_state()
        self.assertEqual(state["pod_id"], "p2")
        self.assertIn("deleted_at", state["history"][0])
        self.assertNotIn("deleted_at", state["history"][1])

    def test_terminate_own_clears_its_own_pod(self):
        rp = FakeRunPod()
        pod.terminate_own(rp, "p2")
        self.assertEqual(rp.deleted, ["p2"])
        self.assertNotIn("pod_id", config.load_state())

    def test_watchdog_of_old_pod_keeps_new_one(self):
        with mock.patch("sys.stdout"):
            watchdog._wd_check(FakeRunPod(error=RunPodError(404, "gone")), "p1", 3600, 0)
        self.assertEqual(config.load_state()["pod_id"], "p2")

    def test_abort_up_of_old_pod_keeps_new_one(self):
        rp = FakeRunPod()
        with mock.patch.object(cli.mascot, "show"), mock.patch("sys.stdout"),                 self.assertRaises(ConfigError):
            cli.abort_up(rp, "p1", ["boom"], keep=False, reason="test")
        self.assertEqual(rp.deleted, ["p1"])
        self.assertEqual(config.load_state()["pod_id"], "p2")


class GuardActivity(unittest.TestCase):
    METRICS = """# HELP vllm:num_requests_running Number of requests
vllm:num_requests_running{model_name="ronin"} 1.0
vllm:num_requests_waiting{model_name="ronin"} 2.0
vllm:request_success_total{finished_reason="stop",model_name="ronin"} 5.0
vllm:request_success_total{finished_reason="length",model_name="ronin"} 3.0
"""

    def test_parses_busy_and_finished(self):
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = self.METRICS.encode()
        with mock.patch.object(guard.urllib.request, "urlopen", return_value=resp):
            self.assertEqual(guard.activity("8000"), (3.0, 8.0))

    def test_sends_the_pod_key(self):
        # llama-server only leaves /health public: /metrics needs the API key
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = self.METRICS.encode()
        with mock.patch.dict(os.environ, {"LLAMA_API_KEY": "k1"}), \
             mock.patch.object(guard.urllib.request, "urlopen", return_value=resp) as urlopen:
            guard.activity("8000")
        self.assertEqual(urlopen.call_args[0][0].get_header("Authorization"), "Bearer k1")

    def test_refused_metrics_are_reported(self):
        err = guard.urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with mock.patch.object(guard, "_refused", False), \
             mock.patch.object(guard.urllib.request, "urlopen", side_effect=err), \
             mock.patch.object(guard, "log") as log:
            self.assertIsNone(guard.activity("8000"))
            self.assertIsNone(guard.activity("8000"))
        log.assert_called_once()      # said once, not every 30 s

    def test_loading_503_is_not_a_refusal(self):
        err = guard.urllib.error.HTTPError("u", 503, "Loading model", {}, None)
        with mock.patch.object(guard, "_refused", False), \
             mock.patch.object(guard.urllib.request, "urlopen", side_effect=err), \
             mock.patch.object(guard, "log") as log:
            self.assertIsNone(guard.activity("8000"))
        log.assert_not_called()

    def test_none_while_server_is_down(self):
        with mock.patch.object(guard.urllib.request, "urlopen", side_effect=OSError("refused")):
            self.assertIsNone(guard.activity("8000"))

    def test_idle_triggers_deletion(self):
        clock = iter(range(0, 10_000, 600))     # each check is 10 minutes later
        with mock.patch.object(guard, "activity", return_value=(0.0, 4.0)), \
             mock.patch.object(guard.time, "monotonic", side_effect=lambda: next(clock)), \
             mock.patch.object(guard.time, "sleep"), \
             mock.patch.object(guard, "log"), \
             mock.patch.object(guard, "delete_self", return_value=True) as delete:
            guard.watch(deadline=0, idle=1800, port="8000")
        delete.assert_called_once()

    def test_deadline_triggers_deletion(self):
        with mock.patch.object(guard.time, "time", return_value=2_000), \
             mock.patch.object(guard.time, "sleep"), \
             mock.patch.object(guard, "log"), \
             mock.patch.object(guard, "delete_self", return_value=True) as delete:
            guard.watch(deadline=1_000, idle=0, port="8000")
        delete.assert_called_once()


class StatusExitCodes(ProjectTestCase):
    def run_status(self):
        args = mock.Mock(json=True)
        with mock.patch("builtins.print"):
            return cli.cmd_status(args, config.load_config())

    def test_no_pod(self):
        self.assertEqual(self.run_status(), cli.STATUS_NO_POD)

    def test_waiting_for_stock_is_not_ready(self):
        now = cli.time.time()
        config.save_state({"waiting": {"profile": "main", "clouds": ["COMMUNITY"],
                                       "since": now - 300, "heartbeat": now - 5}})
        with mock.patch.object(cli, "client", side_effect=ConfigError("no key")):
            self.assertEqual(self.run_status(), cli.STATUS_NOT_READY)

    def test_stale_wait_is_ignored(self):
        old = cli.time.time() - 3600       # that `ronin up` died an hour ago
        config.save_state({"waiting": {"profile": "main", "clouds": ["COMMUNITY"],
                                       "since": old, "heartbeat": old}})
        with mock.patch.object(cli, "client", side_effect=ConfigError("no key")):
            self.assertEqual(self.run_status(), cli.STATUS_NO_POD)

    def test_ready_pod(self):
        config.save_state({"pod_id": "p1"})
        rp = FakeRunPod({"id": "p1", "status": "RUNNING", "runtime": {"uptime": 60}, "cost": 1.0})
        with mock.patch.object(cli, "client", return_value=rp), \
             mock.patch.object(cli, "healthy", return_value=True):
            self.assertEqual(self.run_status(), cli.STATUS_READY)


class Costs(ProjectTestCase):
    def test_estimate_counts_from_creation_not_uptime(self):
        # 10 min since creation, but the container has only been up 2 min (image pull)
        pod = {"runtime": {"uptime": 120}, "cost": 0.60}
        with mock.patch.object(pod_mod.time, "time", return_value=1_000_600):
            text = pod_mod.session_cost(pod, since=1_000_000)
        self.assertIn("$0.10", text)
        self.assertIn("estimate", text)

    def test_real_cost_is_the_balance_drop(self):
        rp = mock.Mock(account=mock.Mock(return_value={"balance": 14.66, "spend_per_hour": 0}))
        self.assertAlmostEqual(pod_mod.real_session_cost(rp, {"balance_at_up": 14.82}), 0.16)
        self.assertIsNone(pod_mod.real_session_cost(rp, {}))      # no balance recorded

    def test_history_survives_the_pod_and_marks_deletion(self):
        state = {}
        pod_mod.remember_pod(state, {"id": "p1", "gpu": {"id": "G"}, "cost": 0.5}, "basic-general")
        state.update(pod_id="p1", api_key="k", created_at=1, balance_at_up=9)
        config.save_state(state)
        pod_mod.terminate(FakeRunPod(), state)
        saved = config.load_state()
        self.assertNotIn("pod_id", saved)
        self.assertNotIn("balance_at_up", saved)
        self.assertEqual(saved["history"][0]["profile"], "basic-general")
        self.assertIn("deleted_at", saved["history"][0])

    def test_estimate_from_history_rate(self):
        state = {"pod_id": "p1", "created_at": 1_000_000,
                 "history": [{"pod_id": "p1", "cost_per_hour": 2.49}]}
        with mock.patch.object(cli.time, "time", return_value=1_000_000 + 449):
            self.assertAlmostEqual(cli.estimated_cost(state), 0.31, places=2)
        self.assertIsNone(cli.estimated_cost({"pod_id": "p1"}))

    def test_history_is_capped(self):
        state = {}
        for i in range(pod_mod.HISTORY_MAX + 5):
            pod_mod.remember_pod(state, {"id": f"p{i}"}, "x")
        self.assertEqual(len(state["history"]), pod_mod.HISTORY_MAX)
        self.assertEqual(state["history"][-1]["pod_id"], f"p{pod_mod.HISTORY_MAX + 4}")

    def test_billing_rows_are_summed_per_pod(self):
        rows = [{"podId": "a", "amount": 1.1, "timeBilledMs": 3_600_000, "time": "2026-10-21 05:00:00"},
                {"podId": "a", "amount": 0.1, "timeBilledMs": 300_000, "time": "2026-10-21 04:00:00"},
                {"podId": "b", "amount": 0.2, "timeBilledMs": 60_000, "time": "2026-10-21 06:00:00"}]
        pods = cli.billed_by_pod(rows)
        self.assertAlmostEqual(pods["a"]["amount"], 1.2)
        self.assertAlmostEqual(pods["a"]["minutes"], 65)
        self.assertEqual(pods["a"]["first"], "2026-10-21 04:00:00")


class Snapshot(unittest.TestCase):
    def wait_done(self, snap):
        for _ in range(200):
            if snap.done:
                return
            cli.time.sleep(0.01)
        self.fail("snapshot never finished")

    def test_reads_lines_and_closes_the_stream(self):
        resp = mock.Mock()
        rp = mock.Mock(open_logs=mock.Mock(return_value=resp),
                       stream_logs=mock.Mock(return_value=iter([{"line": "a"}, {"line": "b"}])))
        snap = pod_mod.LogSnapshot(rp, "p1", wait=5)
        self.wait_done(snap)
        self.assertEqual(snap.lines, ["a", "b"])
        resp.close.assert_called()

    def test_errors_are_swallowed(self):
        rp = mock.Mock(open_logs=mock.Mock(side_effect=RunPodError(0, "offline")))
        snap = pod_mod.LogSnapshot(rp, "p1")
        self.wait_done(snap)
        self.assertEqual(snap.lines, [])


class WaitForStock(unittest.TestCase):
    NO_STOCK = RunPodError(500, "There are no longer any instances available with the requested specifications.")
    BAD = RunPodError(400, "invalid image")

    def run_create(self, side_effect, clouds=("COMMUNITY",), max_wait=0):
        rp = mock.Mock(create_pod=mock.Mock(side_effect=side_effect))
        sleep = mock.Mock()
        cands = [fit.Fit("GPU A", None, None), fit.Fit("GPU B", None, None)]
        with mock.patch("builtins.print"), mock.patch("sys.stdout"):
            result = cli.create_with_stock(rp, mock.Mock(datacenter=""), cands, list(clouds),
                                           lambda gpu, cloud: {"gpu": gpu, "cloud": cloud},
                                           max_wait=max_wait, sleep=sleep)
        return result, rp, sleep

    def test_waits_until_stock_appears(self):
        # round 1: both GPUs out of stock -> wait; round 2: GPU A has stock
        (pod, cand, cloud), rp, sleep = self.run_create([self.NO_STOCK, self.NO_STOCK, {"id": "p"}])
        self.assertEqual((pod["id"], cand.gpu, cloud), ("p", "GPU A", "COMMUNITY"))
        self.assertEqual(sleep.call_count, cli.STOCK_POLL)    # exactly one wait round

    def test_secure_is_tried_after_community_without_waiting(self):
        (pod, cand, cloud), rp, sleep = self.run_create(
            [self.NO_STOCK, self.NO_STOCK, {"id": "p"}], clouds=("COMMUNITY", "SECURE"))
        self.assertEqual(cloud, "SECURE")
        self.assertEqual(rp.create_pod.call_args[0][0], {"gpu": "GPU A", "cloud": "SECURE"})
        sleep.assert_not_called()

    def test_refused_request_does_not_wait(self):
        with self.assertRaises(ConfigError):
            self.run_create([self.BAD, self.BAD])

    def test_max_wait_gives_up(self):
        with mock.patch.object(cli.time, "monotonic", side_effect=[0, 0, 10_000, 10_000]), \
             self.assertRaises(ConfigError):
            self.run_create([self.NO_STOCK] * 10, max_wait=60)


class MemoryFit(unittest.TestCase):
    BF16_80B = {"BF16": 80 * 2 ** 30}
    CATALOG = [{"id": "96GB", "memory": 96}, {"id": "80GB", "memory": 80}]

    def profile(self, *args):
        return config.Profile("x", "org/model", ["80GB", "96GB", "unknown"], list(args))

    def test_bf16_weights(self):
        self.assertAlmostEqual(fit.weights_gib(self.BF16_80B, self.profile()), 160)

    def test_online_fp8_halves_16_bit_weights(self):
        p = self.profile("--quantization", "fp8_per_tensor")
        self.assertAlmostEqual(fit.weights_gib(self.BF16_80B, p), 80)
        self.assertEqual(fit.precision(self.BF16_80B, p), "fp8_per_tensor on load")

    def test_prequantized_checkpoint_counts_its_own_dtypes(self):
        gptq = {"I32": 2 * 2 ** 30, "F16": 2 ** 30}   # packed int4 + 16-bit leftovers
        p = self.profile("--quantization=gptq")
        self.assertAlmostEqual(fit.weights_gib(gptq, p), 3)
        self.assertEqual(fit.precision(gptq, p), "int4")

    def test_fits_per_gpu(self):
        p = self.profile("--quantization", "fp8", "--gpu-memory-utilization", "0.95")
        results = fit.fits(p, 74, self.CATALOG)
        self.assertEqual([f.gpu for f in results], ["80GB", "96GB", "unknown"])
        self.assertFalse(results[0].fits)
        self.assertTrue(results[1].fits and not results[1].tight)
        self.assertTrue(results[2].fits)          # not in the catalog: not judged
        self.assertIsNone(results[2].spare_gib)

    def test_tensor_parallel_pools_vram(self):
        # 150 GiB weights fit on one 80 GB card only when spread over several.
        one = config.Profile("x", "m", ["80GB"], ["--gpu-memory-utilization", "0.95"], gpu_count=1)
        four = config.Profile("x", "m", ["80GB"], ["--gpu-memory-utilization", "0.95"], gpu_count=4)
        self.assertFalse(fit.fits(one, 150, self.CATALOG)[0].fits)
        self.assertTrue(fit.fits(four, 150, self.CATALOG)[0].fits)

    def test_up_stops_before_creating_a_pod(self):
        rp = mock.Mock(list_gpus=mock.Mock(return_value=self.CATALOG))
        with mock.patch.object(fit, "checkpoint_params", return_value=self.BF16_80B), \
             mock.patch("builtins.print"):
            p = config.Profile("x", "org/model", ["80GB", "96GB"], [])
            with self.assertRaises(ConfigError):
                cli.memory_check(rp, mock.Mock(cloud="SECURE"), p)
        rp.create_pod.assert_not_called()

    def test_unknown_size_does_not_block(self):
        rp = mock.Mock(list_gpus=mock.Mock(return_value=self.CATALOG))
        with mock.patch.object(fit, "checkpoint_params", side_effect=LookupError("offline")), \
             mock.patch("builtins.print"):
            p = self.profile()
            self.assertEqual([f.gpu for f in cli.memory_check(rp, mock.Mock(cloud="SECURE"), p)], p.gpus)


if __name__ == "__main__":
    unittest.main()
