"""In-pod guard: run vLLM and delete the pod at a deadline or once it sits idle.

This file is not imported by ronin: its source is sent to the pod and run there as
`python3 -c <source> vllm serve ...` (standard library only, inside the vLLM image).
It works even when your PC is asleep or off, unlike `ronin watchdog`.

Environment (set by `ronin up`):
  RONIN_DEADLINE       unix time after which the pod deletes itself (0 = no deadline)
  RONIN_IDLE           seconds without requests before it deletes itself (0 = never)
  RONIN_PORT           port the server listens on (its /metrics tells requests apart)
  RONIN_BUSY_METRICS   comma-separated gauges of in-flight work (default: vLLM's)
  RONIN_DONE_METRIC    counter that moves on every finished request (default: vLLM's)
RunPod provides RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY in every pod.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

CHECK_EVERY = 30
# Tried in order: delete is what ronin wants; stop at least ends GPU billing.
ACTIONS = (
    ("DELETE", "https://api.runpod.io/v2/pods/{id}", None),
    ("DELETE", "https://rest.runpod.io/v1/pods/{id}", None),
    ("POST", "https://api.runpod.io/v2/pods/{id}/action", {"action": "stop"}),
)
# Gauges of in-flight work and a counter that moves on every finished request. The
# defaults are vLLM's; a non-vLLM backend (e.g. llama.cpp) passes its own names in the env.
BUSY = tuple(m for m in os.environ.get(
    "RONIN_BUSY_METRICS", "vllm:num_requests_running,vllm:num_requests_waiting").split(",") if m)
DONE = os.environ.get("RONIN_DONE_METRIC", "vllm:request_success_total")


def log(msg):
    print(f"[ronin-guard] {msg}", flush=True)


_refused = False


def activity(port):
    """(busy, finished) from the server's /metrics, or None while the server is not up.

    llama-server guards /metrics with its API key (only /health is public), so the key
    the pod already holds is sent; vLLM's /metrics is public and ignores it."""
    global _refused
    headers = {}
    key = os.environ.get("LLAMA_API_KEY") or os.environ.get("VLLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        # 503 is llama-server still loading the model: not up yet, nothing to report.
        # A refusal (401/403) means the idle limit would silently never fire (only the
        # deadline would still protect the pod): say so, once.
        if e.code in (401, 403) and not _refused:
            _refused = True
            log(f"/metrics answered HTTP {e.code}: the idle limit cannot see activity")
        return None
    except Exception:
        return None
    busy = finished = 0.0
    for line in text.splitlines():
        name, _, value = line.rpartition(" ")
        name = name.split("{", 1)[0]
        try:
            if name in BUSY:
                busy += float(value)
            elif name == DONE:
                finished += float(value)
        except ValueError:
            continue
    return busy, finished


def delete_self():
    pod_id, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not key:
        log("RUNPOD_POD_ID / RUNPOD_API_KEY missing: cannot delete the pod")
        return False
    for method, url, body in ACTIONS:
        req = urllib.request.Request(
            url.format(id=pod_id), method=method,
            data=json.dumps(body).encode() if body else None,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "User-Agent": "ronin-guard"})
        try:
            urllib.request.urlopen(req, timeout=30).close()
            log(f"{method} {url.format(id=pod_id)}: done")
            return True
        except Exception as e:
            log(f"{method} {url.format(id=pod_id)} failed: {e}")
    return False


def watch(deadline, idle, port):
    idle_since = None       # monotonic time of the last observed activity
    last_finished = None
    while True:
        reason = None
        if deadline and time.time() >= deadline:
            reason = "deadline reached"
        elif idle:
            sample = activity(port)
            if sample is not None:
                busy, finished = sample
                if idle_since is None or busy or finished != last_finished:
                    idle_since = time.monotonic()
                last_finished = finished
                if time.monotonic() - idle_since >= idle:
                    reason = f"idle for {int(idle // 60)} min"
        if reason:
            log(f"{reason}: deleting the pod so the GPU stops billing")
            if delete_self():
                return
        time.sleep(CHECK_EVERY)


def main():
    deadline = float(os.environ.get("RONIN_DEADLINE") or 0)
    idle = float(os.environ.get("RONIN_IDLE") or 0)
    port = os.environ.get("RONIN_PORT", "8000")
    parts = []
    if deadline:
        parts.append(time.strftime("deletes itself at %H:%M UTC", time.gmtime(deadline)))
    if idle:
        parts.append(f"or after {int(idle // 60)} min idle")
    log(" ".join(parts) or "no limits set")

    child = subprocess.Popen(sys.argv[1:])
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _f: child.send_signal(s))
    if deadline or idle:
        threading.Thread(target=watch, args=(deadline, idle, port), daemon=True).start()
    sys.exit(child.wait())


if __name__ == "__main__":  # also true under `python3 -c`
    main()
