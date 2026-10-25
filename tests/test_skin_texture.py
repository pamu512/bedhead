"""skin_texture dial: how much pore/edge detail survives skin smoothing.

The frequency-separation skin effect attenuates the detail layer by
`att = 1 - (1 - tex_keep) * skin_alpha`. These tests pin the dial's contract:
monotonicity, the tex=1 identity, the tex=0 floor matching the legacy
hardcoded behavior, and JSON persistence.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bedhead.config import Preset
from bedhead.retoucher import _retouch_roi, skin_mask
from bedhead.tracker import FACE_OVAL, FaceFrame

H, W = 240, 208


def _face() -> FaceFrame:
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
    place(LEFT_EYE_RING, -0.16, -0.08, 0.05, 0.025)
    place(RIGHT_EYE_RING, 0.16, -0.08, 0.05, 0.025)
    place(LEFT_BROW, -0.16, -0.15, 0.06, 0.015)
    place(RIGHT_BROW, 0.16, -0.15, 0.06, 0.015)
    place(OUTER_LIPS, 0.0, 0.22, 0.09, 0.045)
    place(INNER_LIPS, 0.0, 0.22, 0.07, 0.03)
    return FaceFrame(landmarks=landmarks, score=1.0, h=H, w=W)


def _roi() -> np.ndarray:
    rng = np.random.default_rng(7)
    roi = np.zeros((H, W, 3), np.uint8)
    roi[:] = (150, 180, 205)
    noise = rng.normal(0, 6.0, (H, W, 1)).repeat(3, axis=2)
    roi = np.clip(roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.circle(roi, (W // 2, H // 2), 28, (90, 120, 160), -1)
    return roi


def _skin_only(tex: float) -> Preset:
    return Preset(
        intensity=1.0, skin=1.0, skin_texture=tex,
        under_eye=0.0, shine=0.0, teeth=0.0, hairline=0.0,
        detail_boost=0.0, vibrance=0.0,
    )


def _lap(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


class TestSkinTextureDial:
    def test_full_keep_is_identity_in_mask_core(self):
        roi, face = _roi(), _face()
        out = _retouch_roi(roi.copy(), face, _skin_only(1.0))
        core = skin_mask(face) >= 254
        assert core.sum() > 400
        assert np.allclose(out[core], roi[core], atol=2), (
            "tex=1.0 must reconstruct the source inside the mask core "
            "(±2 for the 8-bit LAB round-trip)"
        )

    def test_zero_removes_detail_energy(self):
        roi, face = _roi(), _face()
        out = _retouch_roi(roi.copy(), face, _skin_only(0.0))
        assert _lap(out) < _lap(roi) * 0.8, "tex=0.0 must actually smooth"

    def test_dial_monotonic(self):
        roi, face = _roi(), _face()
        v0 = _lap(_retouch_roi(roi.copy(), face, _skin_only(0.0)))
        v5 = _lap(_retouch_roi(roi.copy(), face, _skin_only(0.5)))
        v1 = _lap(_retouch_roi(roi.copy(), face, _skin_only(1.0)))
        assert v0 < v5 < v1

    def test_default_sits_above_legacy_floor_behavior(self):
        # legacy: att floor 0.15 at full alpha. default 0.35 keeps more texture
        # than the old hardcode, never less.
        roi, face = _roi(), _face()
        legacy = Preset(
            intensity=1.0, skin=1.0, skin_texture=0.15,
            under_eye=0.0, shine=0.0, teeth=0.0, hairline=0.0,
            detail_boost=0.0, vibrance=0.0,
        )
        default = _skin_only(Preset().skin_texture)
        assert _lap(_retouch_roi(roi.copy(), face, legacy)) <= _lap(
            _retouch_roi(roi.copy(), face, default)
        )

    def test_json_round_trip(self, tmp_path):
        p = Preset(skin_texture=0.62)
        path = tmp_path / "preset.json"
        p.save(str(path))
        loaded = Preset.load(str(path))
        assert loaded.skin_texture == pytest.approx(0.62)
