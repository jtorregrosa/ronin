"""Unit tests for ronin (standard library only): `python -m unittest`."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ronin import api, catalog, cli, config, fit, guard, pod, runpod, vast, watchdog

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
        # no real provider keys: nothing (the orphan check, say) may reach a live API
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in ("RUNPOD_API_KEY", "VAST_API_KEY"):
            os.environ.pop(key, None)

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
        patcher = mock.patch.object(api.time, "sleep")
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
    name, label = "runpod", "RunPod"

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
        # a top-up meanwhile: unknown, not $0.00
        self.assertIsNone(pod_mod.real_session_cost(rp, {"balance_at_up": 10.0}))

    def down_text(self, session, real, estimate):
        with mock.patch.dict(cli.os.environ, {"NO_COLOR": "1"}):
            return "\n".join(cli.down_cost_lines(session, real, estimate, "RunPod"))

    def test_down_shows_the_real_cost_once_debited(self):
        text = self.down_text("27m · ≈ $0.79 (estimate)", 0.78, 0.79)
        self.assertIn("Real cost: $0.78", text)

    def test_down_never_reports_an_undebited_session_as_free(self):
        # RunPod debits in batches: 6 minutes in, nothing has left the balance yet
        text = self.down_text("6m · ≈ $0.16 (estimate)", 0.0, 0.16)
        self.assertNotIn("$0.00", text)
        self.assertIn("≈ $0.16", text)
        self.assertIn("has not debited it yet", text)
        text = self.down_text(None, 0.14, 0.31)    # pod already gone: the estimate from history
        self.assertIn("Session: ≈ $0.31", text)
        self.assertIn("has debited $0.14 of it so far", text)
        self.assertNotIn("Real cost", text)
        self.assertNotIn("$0.00", self.down_text(None, 0.0, None))

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


class UpCosts(unittest.TestCase):
    """The $/h `up`, `status` and the estimates use is what the pod really bills."""

    def test_runpod_cost_includes_the_container_disk(self):
        # seen live: a $1.69/h GPU with 200 GB of disk -> the account spends $1.718/h
        pod = runpod.with_disk_cost({"cost": 1.69, "disk": 200})
        self.assertAlmostEqual(pod["cost"], 1.7178, places=4)
        self.assertEqual(pod["gpu_cost"], 1.69)
        self.assertEqual(runpod.with_disk_cost(pod)["cost"], pod["cost"])   # not added twice

    def test_hourly_text_breaks_the_price_down(self):
        with mock.patch.dict(cli.os.environ, {"NO_COLOR": "1"}):
            text = cli.hourly_text({"cost": 1.7178, "gpu_cost": 1.69})
        self.assertIn("$1.72/h", text)
        self.assertIn("GPU $1.69 + disk $0.03", text)
        self.assertEqual(cli.hourly_text({}), "$?/h")

    def test_download_is_priced_on_vast_only(self):
        profile = mock.Mock()
        self.assertIsNone(cli.download_text({"cost": 1.0}, profile))     # RunPod: free
        with mock.patch.object(cli.fit, "download_gb", return_value=120.0):
            text = cli.download_text({"download_cost_per_gb": 0.01}, profile)
        self.assertIn("≈ $1.20 once", text)
        with mock.patch.object(cli.fit, "download_gb", side_effect=LookupError("x")):
            self.assertIn("$10.00/TB", cli.download_text({"download_cost_per_gb": 0.01}, profile))


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


class GpuTable(unittest.TestCase):
    def test_every_spelling_has_one_key(self):
        for name in ("RTX 4090", "NVIDIA GeForce RTX 4090", "rtx 4090"):
            self.assertEqual(catalog.key(name), "RTX 4090")
        self.assertEqual(catalog.key("RTX PRO 6000 S"), "RTX PRO 6000 Server")
        self.assertEqual(catalog.key("Some new GPU"), "Some new GPU")

    def test_runpod_gets_its_id_or_the_name_untouched(self):
        self.assertEqual(catalog.provider_gpu("L40S", "runpod"), "NVIDIA L40S")
        self.assertEqual(catalog.provider_gpu("NVIDIA L40S", "runpod"), "NVIDIA L40S")
        self.assertEqual(catalog.provider_gpu("NVIDIA B300", "runpod"), "NVIDIA B300")

    def test_vast_never_gets_a_guessed_name(self):
        self.assertEqual(catalog.provider_gpu("NVIDIA GeForce RTX 4090", "vast"), "RTX 4090")
        with self.assertRaises(ConfigError):
            catalog.provider_gpu("NVIDIA B300", "vast")

    def test_repo_profiles_are_in_the_table(self):
        # every GPU of the shipped ronin.toml can be rented on both providers
        cfg = config.load_config(config.SOURCE_ROOT / "ronin.toml")
        for p in cfg.profiles.values():
            for gpu in p.gpus:
                row = catalog.find(gpu)
                self.assertIsNotNone(row, gpu)
                self.assertTrue(row.runpod and row.vast and row.vast_max_dph, gpu)


class ProviderConfig(ProjectTestCase):
    def write(self, extra):
        (self.root / "ronin.toml").write_text(MINIMAL.replace('[defaults]\n', '[defaults]\n' + extra),
                                              encoding="utf-8")

    def test_default_is_runpod(self):
        self.assertEqual(config.load_config().provider, "runpod")

    def test_vast(self):
        self.write('provider = "vast"\n')
        text = (self.root / "ronin.toml").read_text(encoding="utf-8")
        (self.root / "ronin.toml").write_text(text.replace('"GPU A", "GPU B"', '"RTX 4090", "L40S"'),
                                              encoding="utf-8")
        self.assertEqual(config.load_config().provider, "vast")

    def test_unknown_provider(self):
        self.write('provider = "lambda"\n')
        with self.assertRaises(ConfigError):
            config.load_config()

    def test_vast_has_no_volumes(self):
        self.write('provider = "vast"\nstorage = "volume"\n')
        with self.assertRaises(ConfigError):
            config.load_config()


class VastBody(ProjectTestCase):
    toml = MINIMAL.replace('gpus = ["GPU A", "GPU B"]', 'gpus = ["NVIDIA GeForce RTX 4090", "L40S"]')

    def body(self, gpu="NVIDIA GeForce RTX 4090", cloud="COMMUNITY", **kw):
        cfg = config.load_config()
        return pod.vast_body(cfg, cfg.profile(None), gpu, "secret", cloud=cloud, **kw)

    def test_offer_filters(self):
        offer = self.body()["offer"]
        self.assertEqual(offer["gpu_name"], {"eq": "RTX 4090"})
        self.assertEqual(offer["num_gpus"], {"eq": 1})
        self.assertEqual(offer["verified"], {"eq": True})
        self.assertEqual(offer["dph_base"], {"lte": catalog.find("RTX 4090").vast_max_dph})
        self.assertGreater(offer["gpu_ram"]["gte"], 20000)        # MiB: a 24 GB card
        self.assertNotIn("hosting_type", offer)
        self.assertNotIn("cuda_max_good", offer)                   # vLLM: unfiltered
        self.assertEqual(self.body(cloud="SECURE")["offer"]["hosting_type"], {"eq": 1})

    def test_server_runs_under_the_guard_with_the_port_published(self):
        inst = self.body()["instance"]
        self.assertEqual(inst["runtype"], "args")
        self.assertEqual(inst["entrypoint"], "python3")
        self.assertEqual(inst["args"][0], "-c")
        self.assertIn("def main", inst["args"][1])
        self.assertEqual(inst["args"][2:4], ["vllm", "serve"])
        self.assertEqual(inst["env"]["-p 8000:8000"], "1")
        self.assertEqual(inst["env"]["RONIN_PROVIDER"], "vast")
        self.assertEqual(inst["env"]["RONIN_TUNNEL"], "1")
        self.assertEqual(inst["env"]["VLLM_API_KEY"], "secret")
        self.assertNotIn("secret", inst["args"])
        self.assertEqual(inst["label"], "ronin-main")
        self.assertTrue(inst["cancel_unavail"])

    def test_limits_reach_the_guard(self):
        env = self.body(deadline=1_900_000_000, idle_seconds=600)["instance"]["env"]
        self.assertEqual((env["RONIN_DEADLINE"], env["RONIN_IDLE"]), ("1900000000", "600"))

    def test_multi_gpu_caps_the_whole_offer(self):
        cfg = config.load_config()
        p = config.Profile("tp", "org/m", ["L40S"], [], gpu_count=2)
        offer = pod.vast_body(cfg, p, "L40S", "k", cloud="COMMUNITY")["offer"]
        self.assertEqual(offer["num_gpus"], {"eq": 2})
        self.assertEqual(offer["dph_base"], {"lte": catalog.find("L40S").vast_max_dph * 2})

    def test_llamacpp_needs_cuda_12_8(self):
        cfg = config.load_config()
        p = config.Profile("gg", "o/m-GGUF:Q4_K_M", ["RTX 4090"], [], backend="llamacpp")
        body = pod.vast_body(cfg, p, "RTX 4090", "k", cloud="COMMUNITY")
        self.assertEqual(body["offer"]["cuda_max_good"], {"gte": 12.8})
        self.assertEqual(body["instance"]["entrypoint"], "sh")

    def test_unknown_gpu_is_refused(self):
        with self.assertRaises(ConfigError):
            self.body(gpu="GPU nobody knows")

    def test_runpod_body_translates_short_names(self):
        cfg = config.load_config()
        body = pod.pod_body(cfg, cfg.profile(None), "L40S", None, "k")
        self.assertEqual(body["gpu"]["id"], "NVIDIA L40S")
        self.assertNotIn("RONIN_PROVIDER", body["env"])


class VastClient(unittest.TestCase):
    INSTANCE = {"id": 42, "actual_status": "running", "dph_total": 0.5, "gpu_name": "RTX 4090",
                "public_ipaddr": "1.2.3.4\n", "start_date": 1_000_000.0,
                "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "33526"}]}}
    OFFER = {"id": 7, "gpu_name": "RTX 4090", "dph_base": 0.40, "storage_cost": 0.36,
             "geolocation": "Norway, NO", "inet_down_cost": 0.01}

    def test_instance_in_runpod_shape(self):
        with mock.patch.object(vast.time, "time", return_value=1_000_060.0):
            p = vast.normalize(self.INSTANCE)
        self.assertEqual((p["id"], p["status"], p["cost"]), ("42", "RUNNING", 0.5))
        self.assertEqual(p["gpu"]["id"], "RTX 4090")
        self.assertEqual(p["runtime"]["uptime"], 60)
        self.assertEqual(pod.endpoint(p, 8000), "http://1.2.3.4:33526")
        self.assertIsNone(pod.endpoint(p, 9000))     # that port is not published

    def test_runpod_pods_keep_the_proxy(self):
        # RunPod's pods carry their own "ports" list: not to be read as vast's mapping
        p = {"id": "abc", "status": "RUNNING", "ports": ["8000/http"]}
        self.assertEqual(pod.endpoint(p, 8000), pod.proxy_url("abc", 8000))

    def test_statuses(self):
        for raw, status in (("loading", "CREATED"), (None, "CREATED"), ("exited", "EXITED")):
            self.assertEqual(vast.normalize({**self.INSTANCE, "actual_status": raw})["status"], status)
        self.assertIn("EXITED", pod.DEAD_STATES)

    def test_missing_instance_is_a_404(self):
        v = vast.Vast("k")
        with mock.patch.object(v, "_send", return_value={"instances": None}):
            with self.assertRaises(vast.VastError) as ctx:
                v.get_pod("42")
        self.assertEqual(ctx.exception.status, 404)

    def test_rents_the_first_offer_left(self):
        v = vast.Vast("k")
        taken = vast.VastError(410, "no_such_ask Instance type 7 is no longer available.")
        offers = [self.OFFER, {**self.OFFER, "id": 8, "dph_base": 0.45}]
        with mock.patch.object(v, "search_offers", return_value=offers), \
             mock.patch.object(v, "_request", side_effect=[taken, {"new_contract": 99}]) as req:
            p = v.create_pod({"offer": {}, "instance": {"disk": 72}})
        self.assertEqual(req.call_args[0][:2], ("PUT", "/asks/8/"))
        self.assertEqual(p["id"], "99")
        self.assertAlmostEqual(p["cost"], 0.45 + 0.36 * 72 / 720, places=3)

    def test_no_offers_is_no_stock(self):
        v = vast.Vast("k")
        with mock.patch.object(v, "search_offers", return_value=[]):
            with self.assertRaises(vast.VastError) as ctx:
                v.create_pod({"offer": {}, "instance": {}})
        self.assertTrue(cli.is_no_stock(ctx.exception))

    def test_all_offers_taken_is_no_stock(self):
        v = vast.Vast("k")
        taken = vast.VastError(410, "no_such_ask")
        with mock.patch.object(v, "search_offers", return_value=[self.OFFER]), \
             mock.patch.object(v, "_request", side_effect=[taken]):
            with self.assertRaises(vast.VastError) as ctx:
                v.create_pod({"offer": {}, "instance": {}})
        self.assertTrue(cli.is_no_stock(ctx.exception))

    def test_refused_rent_is_not_no_stock(self):
        v = vast.Vast("k")
        refused = vast.VastError(400, "Invalid args")
        with mock.patch.object(v, "search_offers", return_value=[self.OFFER]), \
             mock.patch.object(v, "_request", side_effect=[refused]):
            with self.assertRaises(vast.VastError) as ctx:
                v.create_pod({"offer": {}, "instance": {}})
        self.assertFalse(cli.is_no_stock(ctx.exception))

    def test_gpu_catalog_groups_offers(self):
        v = vast.Vast("k")
        offers = [{"gpu_name": "RTX 4090", "gpu_ram": 24564, "dph_total": 0.5, "geolocation": "Norway, NO"},
                  {"gpu_name": "RTX 4090", "gpu_ram": 24564, "dph_total": 0.4, "geolocation": "Texas, US"}]
        with mock.patch.object(v, "search_offers", return_value=offers):
            [g] = v.list_gpus("COMMUNITY")
        self.assertEqual((g["id"], g["memory"], g["price"]["community"]), ("RTX 4090", 24, 0.4))
        self.assertEqual([d["id"] for d in g["dataCenters"]], ["NO", "US"])
        # the memory check reads it like RunPod's catalog
        p = config.Profile("x", "m", ["NVIDIA GeForce RTX 4090"], [])
        self.assertEqual(fit.fits(p, 10, [g])[0].vram_gb, 24)


class VastLogs(unittest.TestCase):
    def test_new_lines_only(self):
        self.assertEqual(vast.new_lines(["a", "b", "c"], ["b", "c", "d"]), ["d"])
        self.assertEqual(vast.new_lines(["a", "b"], ["a", "b"]), [])
        self.assertEqual(vast.new_lines([], ["a"]), ["a"])
        self.assertEqual(vast.new_lines(["a"], ["x", "y"]), ["x", "y"])   # window moved on

    def test_poll_yields_each_line_once(self):
        v = vast.Vast("k")
        fetches = [["a", "b"], ["a", "b", "c"]]
        poll = vast.LogPoll(v, "1", tail=1)

        def fetch(pod_id, tail):
            if not fetches:
                poll.close()
                return ["a", "b", "c"]
            return fetches.pop(0)

        with mock.patch.object(v, "fetch_logs", side_effect=fetch), \
             mock.patch.object(vast.time, "sleep"):
            self.assertEqual([e["line"] for e in v.stream_logs("1", resp=poll)], ["b", "c"])


class GuardIdentity(unittest.TestCase):
    def test_runpod_by_default(self):
        with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, clear=True):
            self.assertEqual(guard.identity(), ("runpod", "p", "k"))

    def test_vast(self):
        env = {"RONIN_PROVIDER": "vast", "CONTAINER_ID": "42", "CONTAINER_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(guard.identity(), ("vast", "42", "k"))
        env = {"RONIN_PROVIDER": "vast", "VAST_CONTAINERLABEL": "C.38250", "CONTAINER_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(guard.identity()[1], "38250")

    def test_vast_deletes_its_instance(self):
        env = {"RONIN_PROVIDER": "vast", "CONTAINER_ID": "42", "CONTAINER_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(guard.urllib.request, "urlopen") as urlopen, \
             mock.patch.object(guard, "log"):
            self.assertTrue(guard.delete_self())
        req = urlopen.call_args[0][0]
        self.assertEqual((req.get_method(), req.full_url),
                         ("DELETE", "https://console.vast.ai/api/v0/instances/42/"))
        self.assertEqual(req.get_header("Authorization"), "Bearer k")

    def test_tunnel_url_is_read_from_the_log(self):
        lines = ["noise", "[ronin-guard] tunnel https://abc-def.trycloudflare.com", "more"]
        self.assertEqual(cli.tunnel_url(lines), "https://abc-def.trycloudflare.com")
        self.assertIsNone(cli.tunnel_url(["no tunnel"]))
        self.assertTrue(guard.TUNNEL_URL.search("|  https://abc-def.trycloudflare.com  |"))


class ProviderState(ProjectTestCase):
    def test_old_state_is_runpod(self):
        cfg = config.load_config()
        state = {"pod_id": "p1", "api_key": "k"}
        self.assertEqual(pod.pod_provider(state), "runpod")
        self.assertEqual(pod.pod_url(cfg, state), pod.proxy_url("p1", 8000))

    def test_vast_pod_without_url_yet(self):
        cfg = config.load_config()
        state = {"pod_id": "42", "api_key": "k", "provider": "vast"}
        self.assertIsNone(pod.pod_url(cfg, state))
        with self.assertRaises(ConfigError):
            pod.opencode_config(cfg, state)
        state["url"] = "https://x.trycloudflare.com"
        base = pod.opencode_config(cfg, state)["provider"]["ronin"]["options"]["baseURL"]
        self.assertEqual(base, "https://x.trycloudflare.com/v1")

    def test_down_clears_provider_and_url(self):
        state = {"pod_id": "42", "api_key": "k", "provider": "vast", "url": "http://1.2.3.4:5"}
        config.save_state(state)
        pod.terminate(FakeRunPod(), state)
        saved = config.load_state()
        self.assertNotIn("provider", saved)
        self.assertNotIn("url", saved)

    def test_status_asks_the_pods_provider(self):
        config.save_state({"pod_id": "42", "provider": "vast", "url": "http://1.2.3.4:5"})
        rp = FakeRunPod({"id": "42", "status": "RUNNING", "runtime": {"uptime": 60}, "cost": 0.5})
        with mock.patch.object(cli, "client", return_value=rp) as client, \
             mock.patch.object(cli, "healthy", return_value=True), \
             mock.patch("builtins.print"):
            self.assertEqual(cli.cmd_status(mock.Mock(json=True), config.load_config()), cli.STATUS_READY)
        client.assert_called_with("vast")


class FakeVast:
    name, label = "vast", "vast.ai"

    def __init__(self):
        self.created = []

    def account(self):
        return {"balance": 5.0, "spend_per_hour": 0.0}

    def create_pod(self, body):
        self.created.append(body)
        return {"id": "42", "status": "CREATED", "gpu": {"id": "RTX 4090"}, "cost": 0.4,
                "gpu_cost": 0.38, "where": "Norway, NO", "download_cost_per_gb": 0.01}

    def get_pod(self, pod_id, timeout=None):
        return {"id": "42", "provider": "vast", "status": "RUNNING", "cost": 0.4, "gpu": {"id": "RTX 4090"},
                "runtime": {"uptime": 5}, "public_ip": "1.2.3.4", "ports": {8000: 33526}}


class FakeLogs:
    def __init__(self, *args, **kw):
        pass

    def drain(self):
        return ["[ronin-guard] tunnel https://abc.trycloudflare.com"]


class VastUp(ProjectTestCase):
    toml = MINIMAL.replace('gpus = ["GPU A", "GPU B"]', 'gpus = ["RTX 4090"]') \
                  .replace('[defaults]\n', '[defaults]\nprovider = "vast"\n')

    def run_up(self, healthy):
        rp = FakeVast()
        args = mock.Mock(profile=None, skip_memory_check=True, max_hours=None, idle=None, budget=None,
                         secure=False, max_wait=None, poll=None, keep_on_failure=False,
                         provider=None)
        with mock.patch.object(cli, "client", return_value=rp) as client, \
             mock.patch.object(cli, "healthy", side_effect=healthy), \
             mock.patch.object(cli, "LogStream", FakeLogs), \
             mock.patch.object(cli, "LogSnapshot"), \
             mock.patch.object(cli.time, "sleep"), \
             mock.patch.object(cli.mascot, "show"), \
             mock.patch("builtins.print"), mock.patch("sys.stdout"):
            cli.cmd_up(args, config.load_config())
        client.assert_called_with("vast")
        return rp, config.load_state()

    def test_up_records_the_https_tunnel(self):
        rp, state = self.run_up(lambda url, timeout=10: True)
        self.assertEqual(rp.created[0]["offer"]["gpu_name"], {"eq": "RTX 4090"})
        self.assertEqual((state["pod_id"], state["provider"]), ("42", "vast"))
        self.assertEqual(state["url"], "https://abc.trycloudflare.com")
        self.assertEqual(state["history"][-1]["provider"], "vast")

    def test_up_falls_back_to_plain_http(self):
        # the tunnel URL is logged but never answers
        with mock.patch.object(cli, "TUNNEL_WAIT", 0):
            rp, state = self.run_up(lambda url, timeout=10: url.startswith("http://"))
        self.assertEqual(state["url"], "http://1.2.3.4:33526")


class DownloadSize(unittest.TestCase):
    FILES = (("README.md", 10),
             ("org_Model-Q4_K.gguf", 15), ("org_Model-Q4_K_M.gguf", 17),
             ("IQ3_S/Model-IQ3_S-00001-of-00002.gguf", 55), ("IQ3_S/Model-IQ3_S-00002-of-00002.gguf", 29),
             ("mmproj-Model-f16.gguf", 2), ("mmproj-Model-bf16.gguf", 3), ("mtp-Model-Q4_K_M.gguf", 4),
             ("model-00001-of-00002.safetensors", 20), ("model-00002-of-00002.safetensors", 11))

    def gguf(self, model, args=(), hf_file=None):
        return config.Profile("x", model, ["G"], [], backend="llamacpp",
                              llamacpp_args=list(args), hf_file=hf_file)

    def names(self, profile):
        return [f[0] for f in fit.download_files(profile, self.FILES)]

    def test_quant_is_a_whole_token(self):
        # Q4_K_M, not Q4_K; the MTP head is not part of the model download
        self.assertEqual(self.names(self.gguf("o/m:Q4_K_M", ["--no-mmproj"])), ["org_Model-Q4_K_M.gguf"])
        self.assertEqual(self.names(self.gguf("o/m:Q4_K", ["--no-mmproj"])), ["org_Model-Q4_K.gguf"])

    def test_every_shard(self):
        self.assertEqual(len(self.names(self.gguf("o/m:IQ3_S", ["--no-mmproj"]))), 2)

    def test_default_quant_and_mmproj(self):
        # no quant named: llama-server takes Q4_K_M; without --no-mmproj, the smallest projector
        self.assertEqual(self.names(self.gguf("o/m")), ["org_Model-Q4_K_M.gguf", "mmproj-Model-f16.gguf"])

    def test_exact_file(self):
        self.assertEqual(self.names(self.gguf("o/m", ["--no-mmproj"], hf_file="org_Model-Q4_K.gguf")),
                         ["org_Model-Q4_K.gguf"])

    def test_vllm_downloads_the_safetensors(self):
        p = config.Profile("x", "o/m", ["G"], [])
        with mock.patch.object(fit, "repo_files", return_value=self.FILES):
            self.assertAlmostEqual(fit.download_gb(p), 31 / 1e9)


class Prices(ProjectTestCase):
    toml = MINIMAL.replace('gpus = ["GPU A", "GPU B"]', 'gpus = ["RTX 4090", "L40S"]\ndisk_gb = 72')

    def test_session_total(self):
        q = cli.Quote("RTX 4090", stock="5 offers", gpu_h=0.40, disk_h=0.02, per_gb=0.01)
        self.assertAlmostEqual(q.session(4, 18), 0.42 * 4 + 0.18)
        self.assertAlmostEqual(q.session(4, None), 0.42 * 4)      # unknown size: GPU and disk only

    def test_runpod_quotes(self):
        cfg = config.load_config()
        rp = mock.Mock(list_gpus=mock.Mock(return_value=[
            {"id": "NVIDIA GeForce RTX 4090", "price": {"secure": 0.89}, "availability": "NONE"},
            {"id": "NVIDIA L40S", "price": {"secure": 1.09}, "availability": "HIGH"}]))
        q4090, ql40s = cli.runpod_quotes(rp, cfg, [cfg.profile(None)], "SECURE")["main"]
        self.assertEqual((q4090.stock, q4090.gpu_h, q4090.note), (None, 0.89, "no stock"))
        self.assertEqual((ql40s.stock, ql40s.gpu_h, ql40s.per_gb), ("HIGH", 1.09, 0))
        self.assertAlmostEqual(ql40s.disk_h, 72 * 0.10 / 720)

    def test_vast_quotes_search_once_per_distinct_offer(self):
        cfg = config.load_config()
        other = config.Profile("other", "o/m2", ["RTX 4090", "GPU nobody knows"], [], disk_gb=72)
        rp = mock.Mock(quote=mock.Mock(side_effect=lambda f, disk: None if f["gpu_name"]["eq"] == "L40S"
                                       else {"gpu": 0.4, "disk": 0.03, "download": 0.01, "offers": 50,
                                             "hourly_max": 0.45, "download_max": 0.01}))
        quotes = cli.vast_quotes(rp, cfg, [cfg.profile(None), other], "COMMUNITY")
        self.assertEqual(rp.quote.call_count, 2)          # RTX 4090 is shared by both profiles
        q4090, ql40s = quotes["main"]
        self.assertEqual((q4090.stock, q4090.gpu_h, q4090.per_gb), ("50+ offers", 0.4, 0.01))
        self.assertEqual((ql40s.stock, ql40s.note), (None, "no offers"))
        self.assertEqual(quotes["other"][1].note, "not on vast.ai")

    def test_vast_quote_is_the_offer_up_rents(self):
        v = vast.Vast("k")
        offers = [{"dph_base": 0.40, "storage_cost": 0.36, "inet_down_cost": 0.02},
                  {"dph_base": 0.30, "storage_cost": 0.10, "inet_down_cost": 0.0}]
        with mock.patch.object(v, "search_offers", return_value=offers):
            q = v.quote({}, 72)
        self.assertEqual((q["gpu"], q["download"], q["offers"]), (0.40, 0.02, 2))
        self.assertAlmostEqual(q["disk"], 0.36 * 72 / 720)

    def test_command_prints_every_profile(self):
        cfg = config.load_config()
        rp = mock.Mock(label="RunPod", list_gpus=mock.Mock(return_value=[
            {"id": "NVIDIA L40S", "price": {"secure": 1.09}, "availability": "HIGH"}]))
        args = mock.Mock(provider="runpod", secure=True, profiles=[], hours=2)
        with mock.patch.object(cli, "client", return_value=rp), \
             mock.patch.object(fit, "download_gb", return_value=30.0), \
             mock.patch("builtins.print") as out:
            cli.cmd_prices(args, cfg)
        text = "\n".join(str(c.args[0]) if c.args else "" for c in out.call_args_list)
        self.assertIn("main", text)
        self.assertIn(f"${(1.09 + 72 * 0.10 / 720) * 2:.2f}", text)   # 2 h session, free download


class Integrity(ProjectTestCase):
    """ronin.toml is checked as a whole before anything is built from it."""

    def problems(self, text):
        (self.root / "ronin.toml").write_text(text, encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            config.load_config()
        return str(ctx.exception)

    def test_shipped_configs_are_sound(self):
        config.load_config(config.SOURCE_ROOT / "ronin.toml")
        import tomllib
        self.assertEqual(config.validate(tomllib.loads(config.STARTER_CONFIG)), [])

    def test_wrong_types_are_reported_not_raised(self):
        text = self.problems(MINIMAL + 'disk_gb = "big"\n')
        self.assertIn("disk_gb must be an integer", text)
        text = self.problems(MINIMAL.replace('[defaults]\n', '[defaults]\nport = "80a"\n'))
        self.assertIn("port must be an integer", text)

    def test_typos_suggest_the_key(self):
        text = self.problems(MINIMAL + 'vlm_args = ["--x"]\n')
        self.assertIn("unknown key 'vlm_args' (did you mean 'vllm_args'?)", text)

    def test_key_in_the_wrong_table(self):
        text = self.problems(MINIMAL + "max_hours = 3\n")
        self.assertIn("unknown key 'max_hours' (it belongs in [defaults])", text)

    def test_args_as_one_string(self):
        text = self.problems(MINIMAL.replace(
            'vllm_args = ["--max-model-len", "65536", "--enable-prefix-caching"]',
            'vllm_args = "--max-model-len 65536"'))
        self.assertIn('vllm_args must be a list, e.g. ["--max-model-len", "65536"]', text)

    def test_two_arguments_in_one_item(self):
        text = self.problems(MINIMAL.replace('"--max-model-len", "65536"', '"--max-model-len 65536"'))
        self.assertIn("holds several arguments", text)

    def test_values_and_ranges(self):
        text = self.problems(MINIMAL.replace('[defaults]\n', '[defaults]\ncloud = "secure"\n')
                             + "gpu_count = 0\n")
        self.assertIn("cloud must be 'SECURE' or 'COMMUNITY'", text)
        self.assertIn("gpu_count must be between 1 and 8", text)

    def test_every_problem_at_once(self):
        text = self.problems(MINIMAL + 'disk_gb = "big"\nvlm_args = []\n[pod]\nx = 1\n')
        self.assertIn("3 problems", text)
        self.assertIn("unknown table [pod]", text)

    def test_other_backends_args(self):
        text = self.problems(MINIMAL.replace('model = "org/model"', 'model = "org/model"\nbackend = "llamacpp"'))
        self.assertIn("backend 'llamacpp' ignores vllm_args", text)

    def test_extends_unknown_profile(self):
        text = self.problems(MINIMAL + '\n[profiles.b]\nextends = "nope"\n')
        self.assertIn("extends unknown profile 'nope'", text)

    def test_vast_needs_known_gpus(self):
        text = self.problems(MINIMAL.replace('[defaults]\n', '[defaults]\nprovider = "vast"\n'))
        self.assertIn("'GPU A' is not in ronin's GPU table", text)

    def test_numbers_as_args_and_cuda_as_number(self):
        (self.root / "ronin.toml").write_text(
            MINIMAL.replace('"65536"', "65536") + "min_cuda = 12.8\n", encoding="utf-8")
        p = config.load_config().profile(None)
        self.assertEqual((p.vllm_args[1], p.min_cuda), ("65536", "12.8"))


class BrokenConfigStillStops(ProjectTestCase):
    """A broken ronin.toml must never keep `ronin down` from deleting a billing pod."""

    def test_down_works_with_a_broken_config(self):
        config.save_state({"pod_id": "p1", "api_key": "k"})
        (self.root / "ronin.toml").write_text(MINIMAL + 'disk_gb = "big"\n', encoding="utf-8")
        rp = FakeRunPod({"id": "p1", "status": "RUNNING", "runtime": {"uptime": 60}, "cost": 1.0})
        with mock.patch.object(cli, "client", return_value=rp), \
             mock.patch.object(cli.mascot, "show"), mock.patch("builtins.print"), \
             mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            cli.main(["--config", str(self.root / "ronin.toml"), "down"])
        self.assertEqual(rp.deleted, ["p1"])
        self.assertNotIn("pod_id", config.load_state())

    def test_up_refuses_a_broken_config(self):
        (self.root / "ronin.toml").write_text(MINIMAL + 'disk_gb = "big"\n', encoding="utf-8")
        with mock.patch.object(cli, "client") as client, mock.patch("builtins.print"), \
             mock.patch("sys.stdout"), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.main(["--config", str(self.root / "ronin.toml"), "up"])
        client.assert_not_called()

    def test_unexpected_errors_are_logged_with_a_billing_hint(self):
        config.save_state({"pod_id": "p1", "api_key": "k"})
        with mock.patch.object(cli, "cmd_status", side_effect=KeyError("boom")), \
             mock.patch("builtins.print"), mock.patch("sys.stdout"), mock.patch("sys.stderr"), \
             mock.patch.object(cli.ui, "error") as error, mock.patch.object(cli.ui, "hint") as hint, \
             self.assertRaises(SystemExit):
            cli.main(["--config", str(self.root / "ronin.toml"), "status"])
        self.assertIn("KeyError", error.call_args[0][0])
        self.assertTrue((self.root / ".ronin" / "last-error.log").exists())
        self.assertTrue(any("ronin down" in str(c) for c in hint.call_args_list))


if __name__ == "__main__":
    unittest.main()


class Budget(unittest.TestCase):
    """--budget: the guard's deadline from what the pod costs, priced before creating it."""

    def plan(self, quotes, budget, provider="runpod", gb=0.0, candidates=("GPU A", "GPU B")):
        rp = mock.Mock()
        rp.name = provider
        cfg, profile = mock.Mock(), mock.Mock()
        profile.name = "main"
        fake = mock.Mock(return_value={"main": quotes})
        with mock.patch.object(cli, "runpod_quotes", fake), mock.patch.object(cli, "vast_quotes", fake), \
             mock.patch.object(cli.fit, "download_gb", return_value=gb), mock.patch.object(cli.ui, "warn"):
            return cli.budget_plan(rp, cfg, profile, [fit.Fit(c, None, None) for c in candidates],
                                   ["COMMUNITY"], budget)

    def test_hours_from_gpu_and_disk(self):
        plan = self.plan([cli.Quote("GPU A", gpu_h=1.69, disk_h=0.03)], 5.16)
        self.assertAlmostEqual(plan[("GPU A", "COMMUNITY")]["hours"], 3.0)

    def test_vast_counts_the_download_and_the_dearest_fallback_offer(self):
        q = cli.Quote("GPU A", gpu_h=0.4, disk_h=0.05, per_gb=0.01, hourly_max=0.5, per_gb_max=0.02)
        plan = self.plan([q], 3.0, provider="vast", gb=50)
        self.assertAlmostEqual(plan[("GPU A", "COMMUNITY")]["download"], 1.0)
        self.assertAlmostEqual(plan[("GPU A", "COMMUNITY")]["hours"], 4.0)    # (3 - 1) / 0.5

    def test_gpus_the_budget_cannot_keep_are_skipped(self):
        plan = self.plan([cli.Quote("GPU A", gpu_h=10.0), cli.Quote("GPU B", gpu_h=1.0),
                          cli.Quote("GPU C")], 2.0)
        self.assertEqual(list(plan), [("GPU B", "COMMUNITY")])

    def test_too_small_a_budget_creates_nothing(self):
        with self.assertRaises(ConfigError) as ctx:
            self.plan([cli.Quote("GPU A", gpu_h=10.0)], 1.0)
        self.assertIn("not enough", str(ctx.exception))
        with self.assertRaises(ConfigError):
            self.plan([cli.Quote("GPU A", gpu_h=1.0)], 0)


class Runway(unittest.TestCase):
    def warning(self, balance, spend, deadline_in=None):
        state = {"deadline": cli.time.time() + deadline_in * 3600} if deadline_in else {}
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            return " ".join(cli.runway_warning({"balance": balance, "spend_per_hour": spend}, state))

    def test_warns_before_the_guard_deadline(self):
        # seen live: $5.38 at $1.72/h lasts ~3 h, the guard allowed 7 h more
        text = self.warning(5.38, 1.72, deadline_in=7)
        self.assertIn("lasts ≈ 3h 07m", text)
        self.assertIn("before the guard's deadline", text)

    def test_quiet_when_the_balance_outlasts_the_pod(self):
        self.assertEqual(self.warning(50, 1.72, deadline_in=7), "")
        self.assertEqual(self.warning(5, 0), "")                # nothing billing

    def test_warns_when_little_is_left_without_a_guard(self):
        self.assertIn("lasts ≈ 1h 00m", self.warning(1.72, 1.72))


class Orphans(ProjectTestCase):
    def test_pods_named_by_ronin_but_not_registered(self):
        rp = mock.Mock(list_pods=mock.Mock(return_value=[
            {"id": "mine", "name": "ronin-main"}, {"id": "lost", "name": "ronin-big", "status": "RUNNING"},
            {"id": "x", "name": "someone-else"}]))
        rp.name = "runpod"
        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "k"}), \
             mock.patch.object(pod, "client", return_value=rp):
            found = pod.orphans({"pod_id": "mine", "provider": "runpod"})
        self.assertEqual([p["id"] for _, p in found], ["lost"])

    def test_providers_without_a_key_or_unreachable_are_skipped(self):
        rp = mock.Mock(list_pods=mock.Mock(side_effect=RunPodError(0, "offline")))
        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "k"}), \
             mock.patch.object(pod, "client", return_value=rp):
            self.assertEqual(pod.orphans({}), [])


class VastBilling(unittest.TestCase):
    def test_charges_become_billing_rows_across_pages(self):
        pages = [{"results": [{"type": "instance", "source": "Instance-42", "amount": 0.5,
                               "start": 1_793_000_000, "end": 1_793_003_600}],
                  "next_token": "t"},
                 {"results": [{"type": "instance", "source": "43", "amount": 0.2,
                               "start": 1_793_000_000, "end": 1_793_000_600}]}]
        v = vast.Vast("k")
        with mock.patch.object(v, "_request", side_effect=pages) as req:
            rows = v.pod_billing("2026-10-20T00:00:00Z", bucket="day")
        self.assertEqual([r["podId"] for r in rows], ["42", "43"])
        self.assertEqual(rows[0]["timeBilledMs"], 3_600_000)
        self.assertEqual(req.call_args_list[1].kwargs["query"]["after_token"], "t")
        self.assertAlmostEqual(cli.billed_by_pod(rows)["42"]["amount"], 0.5)
