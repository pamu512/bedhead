"""Shared test setup: headless OpenCV plus one-shot model download.

No test opens a camera or GUI window, and nothing imports ``bedhead.cli``,
``bedhead.panel``, or ``bedhead.sinks``; the suite never touches live devices.
"""

import os

# Must be set before the first ``import cv2`` anywhere in the test session.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402

from bedhead.models import ensure_models  # noqa: E402


@pytest.fixture(autouse=True)
def _no_camera_window_or_device_modules():
    """The suite must not open a camera, a window, or the live entrypoints."""
    import sys

    forbidden = ("bedhead.cli", "bedhead.panel", "bedhead.sinks")
    loaded = [name for name in forbidden if name in sys.modules]
    assert loaded == []
    yield
    loaded = [name for name in forbidden if name in sys.modules]
    assert loaded == []


@pytest.fixture(scope="session")
def models():
    """Ensure the MediaPipe task files exist under ./models/.

    ``ensure_models`` skips entries whose destination file already exists,
    so a warm checkout downloads nothing.
    """
    return ensure_models()
