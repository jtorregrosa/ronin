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
  RONIN_PROVIDER       "runpod" (default) or "vast": how the pod deletes itself
  RONIN_TUNNEL         "1": open a Cloudflare quick tunnel to the port and log its HTTPS URL
RunPod provides RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY in every pod; vast.ai
provides CONTAINER_ID and an instance-scoped CONTAINER_API_KEY.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

CHECK_EVERY = 30
# Per provider, tried in order: delete is what ronin wants; stop at least ends GPU
# billing (a stopped vast.ai instance still bills its disk).
ACTIONS = {
    "runpod": (
        ("DELETE", "https://api.runpod.io/v2/pods/{id}", None),
        ("DELETE", "https://rest.runpod.io/v1/pods/{id}", None),
        ("POST", "https://api.runpod.io/v2/pods/{id}/action", {"action": "stop"}),
    ),
    "vast": (
        ("DELETE", "https://console.vast.ai/api/v0/instances/{id}/", None),
        ("PUT", "https://console.vast.ai/api/v0/instances/{id}/", {"state": "stopped"}),
    ),
}
CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+[.]trycloudflare[.]com")
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


def identity():
    """(provider, pod id, pod-scoped API key) from the env the provider sets."""
    provider = os.environ.get("RONIN_PROVIDER") or "runpod"
    if provider == "vast":
        # VAST_CONTAINERLABEL is "C.<id>": a fallback for the id
        label = os.environ.get("VAST_CONTAINERLABEL", "")
        pod_id = os.environ.get("CONTAINER_ID") or label.partition("C.")[2] or None
        return provider, pod_id, os.environ.get("CONTAINER_API_KEY")
    return provider, os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")


def delete_self():
    provider, pod_id, key = identity()
    if not pod_id or not key:
        log(f"{provider}: the pod id / API key are missing from the env: cannot delete the pod")
        return False
    for method, url, body in ACTIONS.get(provider, ()):
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


def tunnel(port):
    """Download cloudflared and open a quick tunnel to the server: an HTTPS URL for a
    provider that only publishes plain HTTP. Logs the URL (ronin reads it from the log)."""
    path = "/tmp/cloudflared"
    try:
        if not os.path.exists(path):
            req = urllib.request.Request(CLOUDFLARED, headers={"User-Agent": "ronin-guard"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
                f.write(resp.read())
            os.chmod(path, 0o755)
        proc = subprocess.Popen(
            [path, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    except Exception as e:
        log(f"tunnel unavailable: {e}")
        return
    announced = False
    for line in proc.stdout:            # keep reading: a full pipe would stall cloudflared
        match = TUNNEL_URL.search(line)
        if match and not announced:
            announced = True
            log(f"tunnel {match.group(0)}")
    log(f"tunnel closed (cloudflared exited with {proc.wait()})")


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

    if os.environ.get("RONIN_TUNNEL") == "1":
        threading.Thread(target=tunnel, args=(port,), daemon=True).start()
    child = subprocess.Popen(sys.argv[1:])
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _f: child.send_signal(s))
    if deadline or idle:
        threading.Thread(target=watch, args=(deadline, idle, port), daemon=True).start()
    sys.exit(child.wait())


if __name__ == "__main__":  # also true under `python3 -c`
    main()
