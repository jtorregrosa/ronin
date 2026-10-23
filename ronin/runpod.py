"""Minimal client for the RunPod REST API v2 (https://api.runpod.io/v2), plus the two
account reads it lacks: the balance (GraphQL) and per-pod billing (REST v1)."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from . import USER_AGENT

BASE_URL = "https://api.runpod.io/v2"
GRAPHQL_URL = "https://api.runpod.io/graphql"
BILLING_URL = "https://rest.runpod.io/v1/billing/pods"


RETRIES = 3                  # extra attempts for idempotent requests on transient errors
RETRY_BACKOFF = 2.0          # seconds; doubles on each attempt


class RunPodError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"RunPod {status}: {detail}")
        self.status = status
        self.detail = detail

    @property
    def transient(self) -> bool:
        """Network failure (0), rate limit or server error: worth retrying."""
        return self.status in (0, 408, 429) or self.status >= 500


class RunPod:
    def __init__(self, api_key: str, timeout: float = 30):
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None,
                 query: dict | None = None, stream: bool = False,
                 timeout: float | None = None, idempotent: bool | None = None):
        # Only idempotent requests are retried: a repeated POST could create two pods.
        if idempotent is None:
            idempotent = method in ("GET", "DELETE")
        attempts = 1 + (RETRIES if idempotent else 0)
        for attempt in range(attempts):
            try:
                return self._send(method, path, body, query, stream, timeout)
            except RunPodError as e:
                if not e.transient or attempt == attempts - 1:
                    raise
                time.sleep(RETRY_BACKOFF * 2 ** attempt)

    def _send(self, method: str, path: str, body: dict | None, query: dict | None,
              stream: bool, timeout: float | None):
        url = path if path.startswith("https://") else BASE_URL + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Accept": "text/event-stream" if stream else "application/json",
        })
        try:
            resp = urllib.request.urlopen(req, timeout=None if stream else timeout or self.timeout)
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                problem = json.loads(raw)
                detail = problem.get("detail") or problem.get("title") or raw
            except json.JSONDecodeError:
                detail = raw or e.reason
            raise RunPodError(e.code, detail) from None
        except urllib.error.URLError as e:
            raise RunPodError(0, f"could not reach RunPod: {e.reason}") from None
        except (TimeoutError, ConnectionError) as e:
            raise RunPodError(0, f"could not reach RunPod: {e}") from None
        if stream:
            return resp
        try:
            with resp:
                raw = resp.read()
        except (OSError, http.client.HTTPException) as e:
            raise RunPodError(0, f"connection to RunPod dropped: {e}") from None
        return json.loads(raw) if raw else {}

    # --- account ---

    def account(self) -> dict:
        """{'balance': $, 'spend_per_hour': $/h}: the live account balance, which RunPod
        debits as pods run (unlike billing, which is published about an hour later)."""
        query = "query { myself { clientBalance currentSpendPerHr } }"
        data = self._request("POST", GRAPHQL_URL, {"query": query}, idempotent=True)
        me = (data.get("data") or {}).get("myself")
        if not me:
            raise RunPodError(0, f"unexpected account answer: {data.get('errors') or data}")
        return {"balance": float(me["clientBalance"]),
                "spend_per_hour": float(me.get("currentSpendPerHr") or 0)}

    def pod_billing(self, start: str, bucket: str = "hour", pod_id: str | None = None) -> list[dict]:
        """Billed rows ({podId, amount, timeBilledMs, time}) since `start` (ISO UTC).
        RunPod publishes them with a delay: a pod deleted minutes ago is not there yet."""
        query = {"bucketSize": bucket, "startTime": start}
        if pod_id:
            query["podId"] = pod_id
        rows = self._request("GET", BILLING_URL, query=query)
        return rows if isinstance(rows, list) else []

    # --- catalog ---

    def list_gpus(self, cloud: str = "SECURE") -> list[dict]:
        query = {"include": "AVAILABILITY", "product": "POD", "cloud": cloud}
        return self._request("GET", "/catalog/gpus", query=query).get("gpus", [])

    # --- network volumes ---

    def list_volumes(self) -> list[dict]:
        return self._request("GET", "/network-volumes").get("networkVolumes", [])

    def create_volume(self, name: str, size_gb: int, datacenter: str) -> dict:
        return self._request("POST", "/network-volumes",
                             {"name": name, "size": size_gb, "dataCenter": datacenter})

    def delete_volume(self, volume_id: str) -> None:
        self._request("DELETE", f"/network-volumes/{volume_id}")

    # --- pods ---

    def create_pod(self, body: dict) -> dict:
        return self._request("POST", "/pods", body)

    def get_pod(self, pod_id: str, timeout: float | None = None) -> dict:
        return self._request("GET", f"/pods/{pod_id}", timeout=timeout)

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")

    def open_logs(self, pod_id: str, tail: int = 100):
        """The raw Server-Sent Events response of the pod's logs (the caller closes it)."""
        return self._request("GET", f"/pods/{pod_id}/logs", query={"tail": tail}, stream=True)

    def stream_logs(self, pod_id: str, tail: int = 100, resp=None) -> Iterator[dict]:
        """Iterate over the pod's log lines (Server-Sent Events)."""
        resp = resp or self.open_logs(pod_id, tail)
        with resp:
            for raw in resp:
                line = raw.decode(errors="replace").rstrip("\r\n")
                if line.startswith("data:"):
                    try:
                        yield json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
