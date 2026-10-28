"""Minimal client for the vast.ai REST API (https://console.vast.ai/api/v0).

vast.ai is a marketplace: a pod is created by searching offers (bundles) and renting one.
The client answers in the shapes the RunPod client uses -- a pod is
{'id', 'status', 'cost', 'gpu': {'id'}, 'runtime': {'uptime'}} -- so the commands work
on either provider. vast.ai also adds 'public_ip' and 'ports' ({internal: external}):
there is no HTTPS proxy, the model port is published on the host's IP.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timezone

from . import USER_AGENT, catalog
from .api import JsonClient, ProviderError

BASE_URL = "https://console.vast.ai/api/v0"
INSTANCES_V1 = "https://console.vast.ai/api/v1/instances/"


OFFERS_PER_GPU = 3           # cheapest offers tried per GPU before moving on to the next


LOG_WINDOW = 1000            # lines asked for on each log poll
LOG_POLL = 10                # seconds between log polls (vast has no log stream)
LOG_READY_WAIT = 20          # seconds to wait for a log request's result to be uploaded


HOURS_PER_MONTH = 30 * 24    # vast prices storage per GB and month


CHARGES_PAGE = 100           # charges per page (the API's maximum)


class VastError(ProviderError):
    provider = "vast.ai"

    @property
    def no_stock(self) -> bool:
        # 410 no_such_ask: someone else rented the offer between the search and the rent
        return self.status == 410 or "no_such_ask" in (self.detail or "")


class NoOffers(VastError):
    no_stock = True


class Vast(JsonClient):
    BASE_URL = BASE_URL
    name = "vast"
    label = "vast.ai"
    error = VastError

    # --- account ---

    def account(self) -> dict:
        """{'balance': $, 'spend_per_hour': $/h}: the prepaid credit left and what the
        account's running instances cost now."""
        me = self._request("GET", "/users/current/")
        spend = sum(i.get("dph_total") or 0 for i in self.list_instances()
                    if i.get("actual_status") == "running")
        return {"balance": float(me.get("credit") or 0), "spend_per_hour": float(spend)}

    def pod_billing(self, start: str, bucket: str = "hour", pod_id: str | None = None) -> list[dict]:
        """Instance charges since `start` (ISO UTC) in the RunPod billing rows' shape:
        {podId, amount, timeBilledMs, time}. vast.ai groups them per instance and day."""
        since = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
        query = {"select_filters": json.dumps({"day": {"gte": int(since), "lte": int(time.time())},
                                               "type": {"in": ["instance"]}}),
                 "limit": CHARGES_PAGE}
        rows = []
        while True:
            page = self._request("GET", "/charges/", query=query, idempotent=True)
            for charge in page.get("results") or []:
                source = str(charge.get("source") or "").rsplit("-", 1)[-1]  # "Instance-123" or "123"
                if pod_id and source != pod_id:
                    continue
                begin, end = charge.get("start") or since, charge.get("end") or time.time()
                rows.append({"podId": source, "amount": float(charge.get("amount") or 0),
                             "timeBilledMs": max(0, end - begin) * 1000,
                             "time": datetime.fromtimestamp(begin, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")})
            if not page.get("next_token"):
                return rows
            query["after_token"] = page["next_token"]

    # --- offers ---

    def search_offers(self, filters: dict, limit: int = 50) -> list[dict]:
        body = {**filters, "type": "on-demand", "limit": limit, "order": [["dph_total", "asc"]]}
        return self._request("POST", "/bundles/", body, idempotent=True).get("offers", [])

    def quote(self, filters: dict, disk_gb: float, limit: int = 50) -> dict | None:
        """What the offer `up` would rent now costs (the first, as create_pod takes it):
        {'gpu': $/h, 'disk': $/h for `disk_gb`, 'download': $/GB, 'hourly_max' and
        'download_max': the dearest of the offers create_pod may fall back to, 'offers':
        how many match, up to `limit`}. None when nothing matches."""
        offers = self.search_offers(filters, limit=limit)
        if not offers:
            return None
        offer = offers[0]
        return {"gpu": offer["dph_base"],
                "disk": (offer.get("storage_cost") or 0) * disk_gb / HOURS_PER_MONTH,
                "download": offer.get("inet_down_cost") or 0,
                # create_pod may rent any of the first few (if the cheapest is taken meanwhile)
                "hourly_max": max(price(o, disk_gb) for o in offers[:OFFERS_PER_GPU]),
                "download_max": max(o.get("inet_down_cost") or 0 for o in offers[:OFFERS_PER_GPU]),
                "offers": len(offers)}

    def list_gpus(self, cloud: str = "SECURE") -> list[dict]:
        """Offers grouped per GPU, in the RunPod catalog's shape: {'id', 'memory' (GB),
        'price': {cloud: cheapest $/h}, 'availability', 'dataCenters' (countries)}."""
        filters = {**catalog.VAST_BASE_FILTERS, **catalog.VAST_CLOUDS[cloud], "num_gpus": {"eq": 1}}
        groups: dict[str, list[dict]] = {}
        rows: dict[str, catalog.Gpu | None] = {}
        for offer in self.search_offers(filters, limit=1000):
            row = catalog.offer_row(offer["gpu_name"], offer.get("gpu_ram"))
            gpu_id = row.name if row else offer["gpu_name"]
            if not row and catalog.find(gpu_id):
                # a smaller card under a known name (A100 SXM4 40 GB): kept apart from it
                gpu_id += f" ({round(offer['gpu_ram'] / 1024)} GB)"
            rows[gpu_id] = row
            groups.setdefault(gpu_id, []).append(offer)
        gpus = []
        for gpu_id, offers in groups.items():
            count = len(offers)
            countries = sorted({(o.get("geolocation") or "")[-2:] for o in offers} - {""})
            row = rows[gpu_id]
            gpus.append({
                "id": gpu_id,
                # nominal GB, like RunPod's catalog (what fit.py expects); for GPUs ronin
                # does not know, from the MiB CUDA reports
                "memory": row.vram_gb if row else round(min(o["gpu_ram"] for o in offers) / 1024 / 0.99),
                "price": {cloud.lower(): min(o["dph_total"] for o in offers)},
                "availability": "HIGH" if count >= 10 else "MEDIUM" if count >= 3 else "LOW",
                "dataCenters": [{"id": c} for c in countries],
            })
        return gpus

    # --- instances ---

    def create_pod(self, body: dict) -> dict:
        """Rent the cheapest offer matching body['offer'] with body['instance'].
        Raises NoOffers (a no-stock error) when none is left."""
        offers = self.search_offers(body["offer"], limit=OFFERS_PER_GPU)
        if not offers:
            raise NoOffers(404, "no offers match the GPU right now")
        error = None
        for offer in offers:
            try:
                answer = self._request("PUT", f"/asks/{offer['id']}/", body["instance"])
            except VastError as e:
                if not e.no_stock:
                    raise
                error = e
                continue
            return {
                "id": str(answer["new_contract"]),
                "status": "CREATED",
                "gpu": {"id": catalog.offer_key(offer["gpu_name"], offer.get("gpu_ram"))},
                "cost": round(price(offer, body["instance"].get("disk", 0)), 3),
                "gpu_cost": offer["dph_base"],
                "where": offer.get("geolocation"),
                "download_cost_per_gb": offer.get("inet_down_cost") or 0,
            }
        raise NoOffers(410, f"the matching offers were taken meanwhile ({error.detail})")

    def get_pod(self, pod_id: str, timeout: float | None = None) -> dict:
        data = self._request("GET", f"/instances/{pod_id}/", timeout=timeout)
        instance = data.get("instances")
        if not instance:
            # vast answers 200 with no instance for one that does not exist (or is gone)
            raise VastError(404, f"instance {pod_id} not found")
        return normalize(instance)

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/instances/{pod_id}/")

    def list_instances(self) -> list[dict]:
        return self._request("GET", INSTANCES_V1).get("instances") or []

    def list_pods(self) -> list[dict]:
        """Every instance of the account as a pod, with its 'name' (vast's label)."""
        return [{**normalize(i), "name": i.get("label")} for i in self.list_instances()]

    # --- logs ---

    def fetch_logs(self, pod_id: str, tail: int = LOG_WINDOW) -> list[str]:
        """The instance's last `tail` log lines. vast uploads them to a URL on request,
        which answers only once the upload is done."""
        answer = self._request("PUT", f"/instances/request_logs/{pod_id}/",
                               {"tail": str(tail)}, idempotent=True)
        url = answer.get("result_url")
        if not url:
            return []
        deadline = time.monotonic() + LOG_READY_WAIT
        while True:
            # a pre-signed URL: no Authorization header
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read().decode(errors="replace").splitlines()
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if time.monotonic() > deadline:
                    raise VastError(0, f"log upload not ready: {e}") from None
                time.sleep(1)

    def open_logs(self, pod_id: str, tail: int = 100) -> "LogPoll":
        return LogPoll(self, pod_id, tail)

    def stream_logs(self, pod_id: str, tail: int = 100, resp=None) -> Iterator[dict]:
        """Log lines as {'line': ...} events, like RunPod's stream: polled, never ends
        until the LogPoll is closed."""
        resp = resp or self.open_logs(pod_id, tail)
        with resp:
            for line in resp:
                yield {"line": line}


class LogPoll:
    """Polls an instance's logs and yields each line once (closable, like a response)."""

    def __init__(self, client: Vast, pod_id: str, tail: int):
        self.client, self.pod_id, self.tail = client, pod_id, tail
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[str]:
        seen = self.client.fetch_logs(self.pod_id, max(self.tail, LOG_WINDOW))
        yield from (seen[-self.tail:] if self.tail else [])
        while not self.closed:
            for _ in range(LOG_POLL * 2):
                if self.closed:
                    return
                time.sleep(0.5)
            current = self.client.fetch_logs(self.pod_id, LOG_WINDOW)
            yield from new_lines(seen, current)
            seen = current


def new_lines(previous: list[str], current: list[str]) -> list[str]:
    """Lines of `current` (a later window of the same log) not already in `previous`."""
    for k in range(min(len(previous), len(current)), 0, -1):
        if previous[-k:] == current[:k]:
            return current[k:]
    return current


def price(offer: dict, disk_gb: float) -> float:
    """$/h of an offer with `disk_gb` of disk (the search prices a default disk size)."""
    return offer["dph_base"] + (offer.get("storage_cost") or 0) * disk_gb / HOURS_PER_MONTH


def normalize(instance: dict) -> dict:
    """A vast instance in the RunPod pod shape (see the module docstring)."""
    status = catalog.VAST_STATUS.get(instance.get("actual_status"), "CREATED")
    start = instance.get("start_date")
    ports = {}
    for spec, binds in (instance.get("ports") or {}).items():
        internal = spec.split("/", 1)[0]
        if binds and internal.isdigit():
            ports[int(internal)] = int(binds[0]["HostPort"])
    gpu = instance.get("gpu_name")
    return {
        "id": str(instance["id"]),
        "provider": "vast",
        "status": status,
        "cost": instance.get("dph_total"),          # GPU + disk, like create_pod's
        "gpu_cost": instance.get("dph_base"),
        "gpu": {"id": catalog.offer_key(gpu, instance.get("gpu_ram")) if gpu else None},
        "runtime": {"uptime": time.time() - start if start and status == "RUNNING" else None},
        "public_ip": (instance.get("public_ipaddr") or "").strip() or None,
        "ports": ports,
        "message": instance.get("status_msg"),
        "label": instance.get("label"),
    }


def endpoint(pod: dict, port: int) -> str | None:
    """http://ip:port of the model server, once vast has published the port."""
    external = (pod.get("ports") or {}).get(port)
    if pod.get("public_ip") and external:
        return f"http://{pod['public_ip']}:{external}"
    return None
