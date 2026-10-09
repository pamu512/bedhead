"""Pure key-translation helper (no GUI deps, safe to unit-test headless)."""
from __future__ import annotations


def key_from_code(k: int) -> str | None:
    """Map an OpenCV waitKey code to a key string, or None if uninteresting.

    Handles: printable ASCII, space, Esc (27). Returns None for
    modifier/mouse events (negative) and non-character codes.
    """
    if k < 0:
        return None
    if k == 27:
        return "\x1b"  # Esc
    if 32 <= k < 127:
        return chr(k)
    return None
