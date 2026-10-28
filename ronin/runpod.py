"""Minimal client for the RunPod REST API v2 (https://api.runpod.io/v2), plus the two
account reads it lacks: the balance (GraphQL) and per-pod billing (REST v1)."""

from __future__ import annotations

import json
from collections.abc import Iterator

from .api import JsonClient, ProviderError

BASE_URL = "https://api.runpod.io/v2"
GRAPHQL_URL = "https://api.runpod.io/graphql"
BILLING_URL = "https://rest.runpod.io/v1/billing/pods"


# RunPod's wording when the GPU is out of stock (vs. a request it will never accept).
NO_STOCK = ("no longer any instances available", "no instances available",
            "out of stock", "insufficient capacity")

CONTAINER_DISK_PRICE_GB_MONTH = 0.10   # container disk while running, $/GB/month
HOURS_PER_MONTH = 30 * 24


class RunPodError(ProviderError):
    provider = "RunPod"

    @property
    def no_stock(self) -> bool:
        return any(s in (self.detail or "").lower() for s in NO_STOCK)


class RunPod(JsonClient):
    BASE_URL = BASE_URL
    name = "runpod"
    label = "RunPod"
    error = RunPodError

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
        return with_disk_cost(self._request("POST", "/pods", body))

    def get_pod(self, pod_id: str, timeout: float | None = None) -> dict:
        return with_disk_cost(self._request("GET", f"/pods/{pod_id}", timeout=timeout))

    def list_pods(self) -> list[dict]:
        """Every pod of the account (with its 'name')."""
        return [with_disk_cost(p) for p in self._request("GET", "/pods").get("pods") or []]

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


def with_disk_cost(pod: dict) -> dict:
    """RunPod's 'cost' is the GPU's $/h alone; the container disk bills on top (what the
    account's spend per hour shows). 'cost' becomes the total, like vast.ai's, and the
    GPU's part is kept in 'gpu_cost'."""
    cost, disk = pod.get("cost"), pod.get("disk") or pod.get("containerDiskInGb") or 0
    if isinstance(cost, (int, float)) and "gpu_cost" not in pod:
        pod["gpu_cost"] = cost
        pod["cost"] = round(cost + disk * CONTAINER_DISK_PRICE_GB_MONTH / HOURS_PER_MONTH, 4)
    return pod
