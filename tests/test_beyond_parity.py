"""Under-eye v4 + tracker gap-bridging: the two beyond-parity fronts.

under_eye_gap is the only FAILING benchmark metric (live residual 19-69 L*
vs the <10 target; raw deficits 63-82). These tests build a synthetic face
with a measured dark trough and assert the benchmark's own metric closes,
plus the plastic-look guard (cheek ceiling) and the v3->v4 regression
contract.

Tracker bridging addresses the fairness gap (62% lock on the african cohort
vs 87-89% elsewhere; misses are 1-3 frame flickers on dark captures).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bedhead.config import Preset
from bedhead.retoucher import _retouch_roi, apply, skin_mask, under_eye_mask
from bedhead.tracker import FACE_OVAL, FaceFrame

H, W = 480, 416


def _face() -> FaceFrame:
    """Realistic-proportioned face: ~67px eyes so the band has a solid core."""
    from bedhead.tracker import (
        INNER_LIPS, LEFT_BROW, LEFT_EYE_RING,
        OUTER_LIPS, RIGHT_BROW, RIGHT_EYE_RING,
    )

    cx, cy = W / 2, H / 2
    landmarks = np.full((478, 3), (cx, cy, 0.0), np.float32)

    def place(indices, ex, ey, ax, ay):
        n = len(indices)
        for k, idx in enumerate(indices):
            a = 2 * np.pi * k / n
            landmarks[idx, 0] = cx + ex * W + ax * W * np.cos(a)
            landmarks[idx, 1] = cy + ey * H + ay * H * np.sin(a)

    place(FACE_OVAL, 0.0, 0.0, 0.42, 0.46)
    place(LEFT_EYE_RING, -0.16, -0.08, 0.08, 0.04)
    place(RIGHT_EYE_RING, 0.16, -0.08, 0.08, 0.04)
    place(LEFT_BROW, -0.16, -0.15, 0.09, 0.02)
    place(RIGHT_BROW, 0.16, -0.15, 0.09, 0.02)
    place(OUTER_LIPS, 0.0, 0.22, 0.09, 0.045)
    place(INNER_LIPS, 0.0, 0.22, 0.07, 0.03)
    return FaceFrame(landmarks=landmarks, score=1.0, h=H, w=W)


def _dark_trough_roi(deficit: float = 55.0) -> np.ndarray:
    """Even skin + two dark trough bands under the eyes (harsh top light)."""
    roi = np.full((H, W, 3), (150, 180, 205), np.uint8)
    # map deficit L* onto a BGR darkening (approx: -1 L* ~ -1.2 per chan)
    dark = np.clip(np.array([150.0, 180.0, 205.0]) - deficit * 1.2, 0, 255)
    face = _face()
    for ring in range(2):
        ex = -0.16 if ring == 0 else 0.16
        cx = int(W / 2 + ex * W)
        # inside the tear-trough band: ~8-38px below the eye center
        cy = int(H / 2 - 0.08 * H + 24)
        cv2.ellipse(roi, (cx, cy), (30, 16), 0, 0, 360, tuple(dark), -1)
    return roi


def _band_gap(out_or_roi: np.ndarray, face: FaceFrame) -> float:
    """The benchmark's under-eye metric: cheek p60 - band p10 on L*."""
    lab = cv2.cvtColor(out_or_roi, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    band = under_eye_mask(face) > 128
    oval = np.zeros((H, W), np.uint8)
    cv2.fillPoly(oval, [face.poly(FACE_OVAL)], 255)
    cheek = (oval > 128) & ~band
    if band.sum() < 20 or cheek.sum() < 20:
        return 0.0
    return float(np.percentile(lab[cheek], 60) - np.percentile(lab[band], 10))

class TestUnderEyeV4:
    def test_closes_benchmark_gap_on_synthetic_trough(self):
        face = _face()
        roi = _dark_trough_roi(deficit=55.0)
        raw_gap = _band_gap(roi, face)
        assert raw_gap > 40.0, f"synthetic trough must start deep (got {raw_gap:.0f})"
        out = _retouch_roi(roi.copy(), face, Preset(intensity=1.0, under_eye=1.0))
        fixed_gap = _band_gap(out, face)
        assert fixed_gap < 10.0, (
            f"v4 must close the benchmark gap: raw {raw_gap:.0f} -> {fixed_gap:.0f} (target < 10)"
        )

    def test_cheek_ceiling_no_glow(self):
        face = _face()
        roi = _dark_trough_roi(deficit=55.0)
        out = _retouch_roi(roi.copy(), face, Preset(intensity=1.0, under_eye=1.0))
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        band = under_eye_mask(face) > 128
        oval = np.zeros((H, W), np.uint8)
        cv2.fillPoly(oval, [face.poly(FACE_OVAL)], 255)
        cheek = (oval > 128) & ~band
        cheek_p60 = float(np.percentile(lab[cheek], 60))
        assert lab[band].max() <= cheek_p60 + 10.0, (
            "band must never glow above the cheek reference (+8 target, +10 tolerance)"
        )

    def test_strength_dial_monotonic(self):
        face = _face()
        roi = _dark_trough_roi(deficit=55.0)
        gaps = []
        for s in (0.0, 0.5, 1.0):
            out = _retouch_roi(roi.copy(), face, Preset(intensity=1.0, under_eye=s))
            gaps.append(_band_gap(out, face))
        assert gaps[0] > gaps[1] > gaps[2], f"gap must shrink as the dial rises: {gaps}"

    def test_light_trough_untouched_ish(self):
        face = _face()
        roi = _dark_trough_roi(deficit=10.0)  # mild shadow
        before = _band_gap(roi, face)
        out = _retouch_roi(roi.copy(), face, Preset(intensity=0.7, under_eye=0.5))
        after = _band_gap(out, face)
        assert after <= before + 2.0, "mild shadows must not be over-lifted"


class TestTrackerBridging:
    def _tracker(self):
        from bedhead.tracker import FaceTracker

        t = FaceTracker()
        return t

    def test_bridge_holds_last_face_then_expires(self):
        t = self._tracker()
        frame = np.full((H, W, 3), (40, 50, 60), np.uint8)  # no face content
        # prime with a real detection using the synthetic face drawing
        from tests.test_retoucher import synthetic_face_frame

        real = synthetic_face_frame(H, W)
        f1 = t.detect(real, 0)
        assert f1 is not None, "prime detection must succeed"
        # ...then the face disappears (dark frame): bridged frames come back
        b1 = t.detect(frame, 33)
        assert b1 is not None, "first lost frame must bridge"
        assert 0.0 < b1.extras.get("bridged", 0.0) <= 1.0
        for i in range(2, 12):
            b = t.detect(frame, i * 33)
            if i <= 10:
                assert b is not None, f"frame {i} within the window must bridge"
            else:
                assert b is None, "bridge must expire after the window"
        t.close()

    def test_bridge_confidence_ramps_down(self):
        t = self._tracker()
        from tests.test_retoucher import synthetic_face_frame

        real = synthetic_face_frame(H, W)
        t.detect(real, 0)
        confs = []
        for i in range(1, 11):
            f = t.detect(np.full((H, W, 3), (40, 50, 60), np.uint8), i * 33)
            confs.append(f.extras.get("bridged", 0.0))
        assert confs[0] > confs[4] > confs[-1], f"confidence must decay: {confs}"
        t.close()

    def test_redetection_resets_bridge(self):
        t = self._tracker()
        from tests.test_retoucher import synthetic_face_frame

        real = synthetic_face_frame(H, W)
        t.detect(real, 0)
        t.detect(np.full((H, W, 3), (40, 50, 60), np.uint8), 33)  # bridge 1
        f = t.detect(real, 66)  # face back
        assert f is not None and "bridged" not in f.extras
        # and the bridge window is full again for the next gap
        b = t.detect(np.full((H, W, 3), (40, 50, 60), np.uint8), 99)
        assert b is not None
        t.close()

    def test_bridged_frame_fades_effects_via_apply(self):
        from tests.test_retoucher import synthetic_face_frame as sff

        frame = sff(H, W)
        face = _face_with_extras({"bridged": 0.3})
        out_full = apply(frame, face, Preset(intensity=1.0, skin=0.9))
        face_full = _face_with_extras({})
        out_bridged = apply(frame, face, Preset(intensity=1.0, skin=0.9))
        # the bridged run must differ from a full-strength run on a NO-face
        # frame (passthrough) - i.e. effects fade with confidence
        noface = apply(np.full((H, W, 3), (40, 50, 60), np.uint8), None,
                       Preset(intensity=1.0, skin=0.9))
        assert noface is not None
        # intensity actually applied is scaled: bridged 0.3 * 1.0 -> weaker
        d_bridged = float(np.abs(
            out_bridged.astype(np.float32) - frame.astype(np.float32)).mean())
        face_hi = _face_with_extras({"bridged": 0.9})
        out_hi = apply(frame, face_hi, Preset(intensity=1.0, skin=0.9))
        d_hi = float(np.abs(
            out_hi.astype(np.float32) - frame.astype(np.float32)).mean())
        assert d_hi > d_bridged, "higher bridge confidence must apply stronger effects"


def _face_with_extras(extras: dict) -> FaceFrame:
    f = _face()
    f.extras = dict(extras)
    return f
