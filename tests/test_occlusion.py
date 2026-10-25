"""Occlusion hardening: a hand on the face must not be retouched.

The synthetic hand test (4 skin-toned fingers + palm over the cheek/jaw,
the classic thinking pose) pins three behaviors:

  1. face_skin_mask subtracts occluder-class pixels (body-skin + others)
     inside the face oval -- the gate drops where the hand is
  2. the full retoucher.apply passes hand pixels through essentially
     untouched (no smoothing painted over fingers)
  3. tracking survives occlusion (mesh still detected) -- pre-existing
     behavior, pinned here as a regression guard

Measured before the fix (2026-10-25, synthetic hand): the hand zone read
55% face-skin / 41% others in the category mask -- the model assigns most
of a hand-on-cheek to "others", with a large minority misrouted to
face-skin, which the old matte would have retouched straight through.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bedhead.config import Preset

pytest.importorskip("mediapipe")

from bedhead.segmenter import Segmenter
from bedhead.tracker import FaceTracker

W, H = 1280, 720
PORTRAIT = (__file__.rsplit("/", 2)[0]
            + "/../one-piece-looks/apps/client/public/fixtures/face-portrait.jpg")


def _frame() -> np.ndarray:
    img = cv2.imread(PORTRAIT)
    if img is None:
        pytest.skip("portrait fixture not available")
    return cv2.resize(img, (W, H))


# Module-level availability check: tests construct MediaPipe sessions
# (FaceTracker/Segmenter) BEFORE _frame() runs, and an unclosed landmarker
# whose test skipped DEADLOCKS MediaPipe's dispatcher during pytest's GC
# teardown (measured: suite exit hangs forever; the isolated test skips
# fine). When the fixture is missing, skip at collection instead.
_FRAME_AVAILABLE = cv2.imread(PORTRAIT) is not None

pytestmark = pytest.mark.skipif(
    not _FRAME_AVAILABLE, reason="portrait fixture not available"
)


def _hand_over_cheek(frame: np.ndarray, face) -> np.ndarray:
    """Realistic thinking-pose hand: separated fingers, nails, knuckle shading."""
    out = frame.copy()
    pts = face.landmarks[:, :2]
    cheek_x = float(pts[234, 0]) + 0.35 * (float(pts[332, 0]) - float(pts[234, 0]))
    cheek_y = float(pts[205, 1])
    base = (192, 167, 147)   # BGR hand skin, slightly darker/cooler than face
    for i in range(4):
        x = int(cheek_x + i * 30 - 12)          # 30px pitch -> ~8px gaps
        y = int(cheek_y + i * 9)
        ang = -25 + i * 8
        cv2.ellipse(out, (x, y), (11, 44), ang, 0, 360, base, -1)
        # knuckle shading band mid-finger
        kx = x - int(18 * np.sin(np.radians(ang)))
        ky = y - 6
        cv2.ellipse(out, (kx, ky), (9, 6), ang, 0, 360, (172, 148, 128), -1, cv2.LINE_AA)
        # nail near the tip (lighter, glossy)
        nx = x + int(34 * np.cos(np.radians(ang + 90)))
        ny = y + int(34 * np.sin(np.radians(ang + 90)))
        cv2.ellipse(out, (nx, ny), (7, 5), ang, 0, 360, (215, 198, 186), -1, cv2.LINE_AA)
    # palm heel below
    cv2.ellipse(out, (int(cheek_x) - 44, int(cheek_y) + 66), (42, 32), 30, 0, 360,
                (186, 161, 141), -1)
    cv2.ellipse(out, (int(cheek_x) - 44, int(cheek_y) + 72), (30, 20), 30, 0, 360,
                (176, 152, 133), -1)
    return out


def _hand_zone_mask(face) -> np.ndarray:
    pts = face.landmarks[:, :2]
    cheek_x = float(pts[234, 0]) + 0.35 * (float(pts[332, 0]) - float(pts[234, 0]))
    cheek_y = float(pts[205, 1])
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cheek_x), int(cheek_y)), (70, 90), 0, 0, 360, 255, -1)
    return m > 128


class TestHandOcclusion:
    def test_tracking_survives_hand(self):
        tr = FaceTracker()
        face = tr.detect(_frame(), 0)
        assert face is not None
        occluded = _hand_over_cheek(_frame(), face)
        face2 = tr.detect(occluded, 33)
        tr.close()
        assert face2 is not None, "mesh must survive a hand on the cheek"

    def test_occluder_subtracted_from_skin_matte(self):
        tr = FaceTracker()
        frame = _frame()
        face = tr.detect(frame, 0)
        occluded = _hand_over_cheek(frame, face)
        seg = Segmenter()
        seg.tick(frame, 0)                    # clean baseline
        clean = seg.face_skin_mask()
        # tick past the refresh interval so the hand frame is actually
        # re-segmented (the model runs every `interval`-th frame)
        for i in range(seg.interval + 1):
            seg.tick(occluded, 33 + i)
        occl = seg.face_skin_mask()
        seg.close(); tr.close()
        hz = _hand_zone_mask(face)
        assert hz.sum() > 5000
        drop = float(clean[hz].mean() - occl[hz].mean())
        assert drop > 0.15, (
            f"hand zone matte must drop with the hand present (drop={drop:.3f})"
        )

    def test_hand_pixels_survive_retouch(self):
        from bedhead.retoucher import apply

        tr = FaceTracker()
        frame = _frame()
        face = tr.detect(frame, 0)
        occluded = _hand_over_cheek(frame, face)
        face2 = tr.detect(occluded, 33)
        seg = Segmenter()
        for i in range(seg.interval * 2 + 2):  # extra ticks: let the EMA settle
            seg.tick(occluded, 33 + i)
        seg_occl = seg.face_skin_mask()
        preset = Preset(intensity=1.0, skin=1.0, under_eye=1.0, shine=1.0)
        out = apply(occluded, face2, preset, segmenter=seg)
        delta = np.abs(out.astype(np.float32) - occluded.astype(np.float32))
        seg.close(); tr.close()
        hz = _hand_zone_mask(face)
        # measure by matte class, not zone average: finger GAPS are legitimately
        # retouched skin, so the zone mean mixes both. Hand core = retouch gate
        # LOW (occluder subtracted); open skin = gate HIGH (forehead control).
        hand_core = hz & (seg_occl < 0.2)
        pts = face.landmarks[:, :2]
        fh = np.zeros((H, W), np.uint8)
        cv2.ellipse(fh, (int(pts[10, 0]), int(pts[10, 1]) + 60), (70, 90), 0, 0, 360, 255, -1)
        open_skin = (fh > 128) & (seg_occl > 0.6)
        assert hand_core.sum() > 2000, f"hand core must be substantial ({int(hand_core.sum())} px)"
        assert open_skin.sum() > 2000
        hand_delta = float(delta[hand_core].mean())
        skin_delta = float(delta[open_skin].mean())
        assert hand_delta < skin_delta * 0.45, (
            f"hand core ({hand_delta:.2f}) must be retouched far less than bare "
            f"skin ({skin_delta:.2f})"
        )
