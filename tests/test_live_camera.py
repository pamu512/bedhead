"""CI-only: run the full pipeline against the real camera for ~8 seconds.

Marked xfail_strict=False with skip conditions so local runs without a
camera are clean; on CI with no camera at all this skips. Run explicitly
with `pytest -m live_camera` on a machine with a webcam to prove the
end-to-end live path (this is the 'takes a live feed' acceptance test).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from bedhead.config import Preset
from bedhead.retoucher import apply
from bedhead.tracker import FaceTracker

pytestmark = pytest.mark.live_camera


def _camera_available() -> bool:
    cap = cv2.VideoCapture(0)
    ok, _ = cap.read()
    cap.release()
    return ok


@pytest.mark.skipif(not _camera_available(), reason="no camera 0 available")
def test_live_camera_pipeline_sustains_realtime() -> None:
    W, H = 1280, 720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    tracker = FaceTracker()
    preset = Preset()
    frames = 0
    faces = 0
    ms: list[float] = []
    try:
        t_end = time.time() + 8.0
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                continue
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            t0 = time.perf_counter()
            face = tracker.detect(frame, frames * 33)
            apply(frame, face, preset)
            ms.append((time.perf_counter() - t0) * 1000)
            frames += 1
            faces += face is not None
    finally:
        cap.release()
        tracker.close()

    assert frames > 8 * 10, f"camera starved the loop: only {frames} frames in 8 s"
    p95 = float(np.percentile(ms, 95))
    assert p95 < 1000 / 24, f"p95 track+retouch {p95:.1f} ms cannot sustain 24 fps"
    # Face presence is scene-dependent (nobody may be at the desk); the
    # passthrough path is covered by headless tests. This test proves the
    # camera -> track -> retouch loop sustains realtime on live frames.
