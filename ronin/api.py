"""What the provider clients (RunPod, vast.ai) share: the error type and a JSON-over-HTTPS
client that retries idempotent requests on transient errors."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from . import USER_AGENT

RETRIES = 3                  # extra attempts for idempotent requests on transient errors
RETRY_BACKOFF = 2.0          # seconds; doubles on each attempt


class ProviderError(Exception):
    provider = "provider"    # set by each client's subclass ("RunPod", "vast.ai")
    no_stock = False         # True when the GPU is simply not available right now

    def __init__(self, status: int, detail: str):
        super().__init__(f"{self.provider} {status}: {detail}")
        self.status = status
        self.detail = detail

    @property
    def transient(self) -> bool:
        """Network failure (0), rate limit or server error: worth retrying."""
        return self.status in (0, 408, 429) or self.status >= 500


class JsonClient:
    """Bearer-authenticated JSON client. Subclasses set BASE_URL, label and error."""

    BASE_URL = ""
    label = "provider"
    error = ProviderError

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
            except ProviderError as e:
                if not e.transient or attempt == attempts - 1:
                    raise
                time.sleep(RETRY_BACKOFF * 2 ** attempt)

    def _send(self, method: str, path: str, body: dict | None, query: dict | None,
              stream: bool, timeout: float | None):
        url = path if path.startswith("https://") else self.BASE_URL + path
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
                detail = (problem.get("detail") or problem.get("msg") or problem.get("title")
                          or problem.get("error") or raw)
            except (json.JSONDecodeError, AttributeError):
                detail = raw or e.reason
            raise self.error(e.code, detail) from None
        except urllib.error.URLError as e:
            raise self.error(0, f"could not reach {self.label}: {e.reason}") from None
        except (TimeoutError, ConnectionError) as e:
            raise self.error(0, f"could not reach {self.label}: {e}") from None
        if stream:
            return resp
        try:
            with resp:
                raw = resp.read()
        except (OSError, http.client.HTTPException) as e:
            raise self.error(0, f"connection to {self.label} dropped: {e}") from None
        return json.loads(raw) if raw else {}
