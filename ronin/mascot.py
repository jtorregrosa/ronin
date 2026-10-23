"""ronin's mascot: a compact kasa-wearing samurai drawn with block characters.

Each pod state has its own pose (eyes and katana). Disabled with RONIN_NO_MASCOT=1,
and skipped automatically when the console cannot render the characters.
"""

from __future__ import annotations

import os
import sys

from . import ui

KASA = ["  ▗▄███▄▖", " ▀▀▀▀▀▀▀▀▀"]
BODY = " ▝▜█████▛▘"
FEET = "   ▘▘ ▝▝"

# eyes and katana per state
POSES = {
    "ready": ("  ▐▛███▜▌", "━━╸"),    # steady gaze, katana drawn
    "wait":  ("  ▐▄███▄▌", ""),       # narrowed eyes: meditating
    "fail":  ("  ▐▙███▟▌", "━ ╸"),    # downcast, broken katana
    "down":  ("  ▐▛███▜▌", ""),       # katana sheathed
    "rest":  ("  ▐▀███▀▌", ""),       # closed eyes: no pod
}


def _art(pose: str) -> list[str]:
    face, katana = POSES[pose]
    return [*KASA, face, BODY + katana, FEET]


def _can_render() -> bool:
    if os.environ.get("RONIN_NO_MASCOT"):
        return False
    sample = "".join(KASA + [BODY, FEET] + [f + k for f, k in POSES.values()])
    try:
        sample.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def show(pose: str, lines: list[str], top: bool = False) -> None:
    """Show the mascot with text to its right (from the hat brim, or from the top
    with top=True). If it cannot be drawn, print the text only.
    """
    lines = [ui.inline(line) for line in lines]
    if not _can_render():
        for line in lines:
            print(f"  {line}" if line else "")
        return
    art = _art(pose)
    # same text column for every pose (the longest katana wins)
    width = max(len(row) for p in POSES for row in _art(p))
    offset = 0 if top else 1
    for i, row in enumerate(art):
        j = i - offset
        text = lines[j] if 0 <= j < len(lines) else ""
        print(f"  {row:<{width}}  {text}".rstrip())
    for extra in lines[len(art) - offset:]:
        print(f"  {'':<{width}}  {extra}".rstrip())
