"""`ronin watchdog`: a local guard that deletes the pod at a total-uptime limit."""

from __future__ import annotations

import sys
import time

from . import mascot, ui
from .config import Config, ConfigError, load_state
from .api import ProviderError
from .pod import DEAD_STATES, forget_own, pod_client, session_cost, terminate_own

WATCHDOG_POLL = 60               # seconds between provider checks while guarding


WATCHDOG_FINAL_POLL = 5          # tighter polling once the deadline is near


def cmd_watchdog(args, cfg: Config) -> None:
    """Guard the active pod and delete it once it has billed for --max-hours.

    Runs in the foreground (like `ronin logs`): leave it in a spare terminal and a
    forgotten session can no longer bill indefinitely. The limit counts the pod's
    total uptime, so restarting the watchdog does not reset the clock.
    """
    max_hours = cfg.max_hours if args.max_hours is None else args.max_hours
    if max_hours <= 0:
        raise ConfigError("--max-hours must be greater than 0\n"
                          "Pass it explicitly, or set max_hours in ronin.toml.")
    poll = max(1, args.poll)
    limit = max_hours * 3600

    state = load_state()
    if not state.get("pod_id"):
        ui.warn("No pod registered.")
        ui.hint("`ronin up` to start one.")
        return
    rp = pod_client(state)
    pod_id = state["pod_id"]

    def fetch() -> dict | None:
        """The pod, or None if it is already gone (404)."""
        try:
            return rp.get_pod(pod_id)
        except ProviderError as e:
            if e.status == 404:
                return None
            raise

    pod = fetch()
    if pod is None:
        ui.warn(f"Pod {pod_id} no longer exists on {rp.label}.")
        ui.hint("`ronin down` to clear the local state.")
        return

    ui.step(f"Guarding pod {pod_id} {ui.dim('· limit ' + ui.duration(limit) + ' total uptime · Ctrl+C to stop')}")
    print()

    frame = 0
    width = 0
    last_poll = time.monotonic()  # monotonic time of the last successful provider check
    base_uptime = (pod.get("runtime") or {}).get("uptime") or 0  # uptime read then
    unreachable = None            # error text while the provider cannot be reached
    try:
        while True:
            now = time.monotonic()
            est_uptime = base_uptime + now - last_poll
            remaining = limit - est_uptime
            # Poll on the interval, or tightly once the deadline is close or the provider failed.
            tight = remaining <= WATCHDOG_POLL or unreachable
            interval = WATCHDOG_FINAL_POLL if tight else poll
            polled = now - last_poll >= interval
            if polled:
                try:
                    fresh = _wd_check(rp, pod_id, limit, width)
                except ProviderError as e:
                    if not e.transient:
                        raise
                    # A network blip must not kill the guard: keep estimating and retry.
                    unreachable = e.detail
                else:
                    if fresh is None:
                        return
                    unreachable = None
                    pod = fresh
                    base_uptime = (pod.get("runtime") or {}).get("uptime") or 0
                    last_poll = time.monotonic()
                    est_uptime = base_uptime
                    remaining = limit - est_uptime

            width = _wd_render(frame, remaining, est_uptime, pod, width, polled, unreachable, rp.label)
            frame += 1
            time.sleep(1)
    except KeyboardInterrupt:
        _wd_clear(width)
        raise


def _wd_check(rp, pod_id: str, limit: float, width: int) -> dict | None:
    """Poll the pod and act on it. Returns the pod, or None once there is nothing
    left to guard (gone, dead and deleted, or deleted at the limit)."""
    try:
        pod = rp.get_pod(pod_id)
    except ProviderError as e:
        if e.status != 404:
            raise
        _wd_clear(width)
        ui.ok("Pod is gone already — nothing left to guard.")
        forget_own(pod_id)
        return None
    if pod["status"] in DEAD_STATES:
        _wd_clear(width)
        terminate_own(rp, pod_id)
        ui.warn(f"Pod was {pod['status']}; deleted it so it cannot be restarted or bill for storage.")
        return None
    uptime = (pod.get("runtime") or {}).get("uptime") or 0
    if uptime < limit:
        return pod
    _wd_clear(width)
    session = session_cost(pod)
    terminate_own(rp, pod_id)
    lines = [f"Reached the {ui.duration(limit)} limit.",
             f"Pod {pod_id} deleted so the GPU stops billing."]
    if session:
        lines.append(ui.dim(f"Session: {session}"))
    lines.append("The ronin sheathes the katana.")
    mascot.show("down", lines)
    return None


def _wd_clear(width: int) -> None:
    if ui.TTY and width:
        sys.stdout.write("\r" + " " * width + "\r")
        sys.stdout.flush()


def _wd_render(frame: int, remaining: float, uptime: float, pod: dict, width: int,
               polled: bool, unreachable: str | None, label: str = "RunPod") -> int:
    """Draw the live countdown line (TTY) or one plain line per check otherwise."""
    left = ui.duration(max(0, remaining))
    cost = pod.get("cost")
    spent = f" · ≈ ${cost * uptime / 3600:.2f} so far" if isinstance(cost, (int, float)) else ""
    if not ui.TTY:
        if polled and unreachable:
            ui.warn(f"{label} unreachable, retrying: {unreachable}")
        elif polled:
            ui.step(f"{left} before the pod is deleted (up {ui.duration(uptime)}{spent})")
        return 0
    spin = ui.FRAMES[frame % len(ui.FRAMES)]
    note = f" · {label} unreachable, retrying" if unreachable else ""
    plain = f"  {spin} {left} left before deletion · up {ui.duration(uptime)}{spent}{note}"
    line = (f"  {ui.cyan(spin)} {ui.bold(left)} left before deletion "
            f"{ui.dim('· up ' + ui.duration(uptime) + spent)}{ui.yellow(note)}")
    pad = " " * max(0, width - len(plain))
    sys.stdout.write("\r" + line + pad)
    sys.stdout.flush()
    return len(plain)
