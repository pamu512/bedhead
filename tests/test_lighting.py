"""Tests for the segmentation-driven effects (studio light, eye light, background).

The real MediaPipe segmenter needs its model file (downloads on first use);
tests construct the Segmenter for real when the model is cached, and use a
synthetic mask path via the lighting functions directly otherwise. On CI the
model download happens on first test run and is cached afterwards.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bedhead.config import Preset
from bedhead.lighting import background_mode, eye_light, studio_light
from bedhead.retoucher import apply
from bedhead.tracker import FaceFrame

W, H = 640, 360


def _frame(w: int = W, h: int = H) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        base[:, :, c] = np.linspace(50 + 40 * c, 150 + 30 * c, w, dtype=np.float32)[None, :]
    # textured: noise + blobs, so blurring measurably changes pixels
    noise = rng.normal(0, 18, (h, w, 1)).repeat(3, axis=2)
    yy = np.linspace(0, 40, h, dtype=np.float32)[:, None]
    return np.clip(base + noise + yy[..., None], 0, 255).astype(np.uint8)


def _person_mask(w: int = W, h: int = H) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    m[:, w // 3: 2 * w // 3] = 1.0  # vertical band as the "person"
    return m


def _face(seed: int = 0) -> FaceFrame:
    rng = np.random.default_rng(seed)
    lm = np.zeros((478, 3), np.float32)
    cx, cy, rx, ry = W / 2, H / 2, W * 0.18, H * 0.36
    for i in range(478):
        t = 2 * np.pi * i / 478
        lm[i, :2] = (cx + rx * np.cos(t), cy + ry * np.sin(t))
    lm[:, 0] = np.clip(lm[:, 0] + rng.normal(0, 1.5, 478), 5, W - 6)
    lm[:, 1] = np.clip(lm[:, 1] + rng.normal(0, 1.5, 478), 5, H - 6)
    return FaceFrame(landmarks=lm, h=H, w=W, extras={"jaw_open": 0.8})


# ---------------------------------------------------------------- studio light


def test_studio_light_brightens_person_only() -> None:
    f = _frame()
    m = _person_mask()
    out = studio_light(f, m, 1.0)
    # person region (mask==1) must brighten
    assert out[H // 2, W // 2].astype(int).sum() >= f[H // 2, W // 2].astype(int).sum()
    # background region must be nearly unchanged
    bg_in = f[H // 2, 20].astype(int)
    bg_out = out[H // 2, 20].astype(int)
    assert abs(int(bg_out.sum()) - int(bg_in.sum())) <= 12


def test_studio_light_zero_noop() -> None:
    f = _frame()
    assert studio_light(f, _person_mask(), 0.0) is f


# ---------------------------------------------------------------- eye light


def test_eye_light_brightens_eye_region() -> None:
    f = _frame()
    face = _face()
    out = eye_light(f, face.landmarks, 1.0)
    # center between the eyes should brighten
    assert int(out[H // 2, W // 2].sum()) >= int(f[H // 2, W // 2].sum())


def test_eye_light_no_landmarks_noop() -> None:
    f = _frame()
    assert eye_light(f, None, 1.0) is f


# ---------------------------------------------------------------- background


def test_background_blur_preserves_person_pixels() -> None:
    f = _frame()
    m = _person_mask()
    out = background_mode(f, m, 1.0, "blur")
    # deep inside the person band the pixel should be ~unchanged (feathered)
    assert abs(int(out[H // 2, W // 2].sum()) - int(f[H // 2, W // 2].sum())) <= 6
    # background must differ (blurred)
    assert abs(int(out[H // 2, 20].sum()) - int(f[H // 2, 20].sum())) > 6


def test_background_dark_darkens_background() -> None:
    f = _frame()
    m = _person_mask()
    out = background_mode(f, m, 1.0, "dark")
    assert int(out[H // 2, 20].sum()) < int(f[H // 2, 20].sum())
    assert abs(int(out[H // 2, W // 2].sum()) - int(f[H // 2, W // 2].sum())) <= 6


def test_background_zero_strength_noop() -> None:
    f = _frame()
    assert background_mode(f, _person_mask(), 0.0, "blur") is f
    assert background_mode(f, _person_mask(), 1.0, "off") is f


# ---------------------------------------------------------------- apply integration


def test_apply_with_seg_effects_runs_without_face() -> None:
    """Background effects must still work when no face is tracked."""
    f = _frame()

    class FakeSeg:
        def person_mask(self, feather: int = 31) -> np.ndarray:
            return _person_mask()

    p = Preset(intensity=0.0, background_strength=1.0, background_mode="blur")
    out = apply(f, None, p, segmenter=FakeSeg())  # type: ignore[arg-type]
    assert out.shape == f.shape
    assert abs(int(out[H // 2, 20].sum()) - int(f[H // 2, 20].sum())) > 6


def test_apply_seg_effects_without_segmenter_noop() -> None:
    """No segmenter -> segmentation effects silently skipped (degrade)."""
    f = _frame()
    p = Preset(intensity=0.0, background_strength=1.0, background_mode="blur")
    out = apply(f, None, p)
    assert out is f


def test_apply_studio_via_named_preset_shapes() -> None:
    f = _frame()

    class FakeSeg:
        def person_mask(self, feather: int = 31) -> np.ndarray:
            return _person_mask()

    p = Preset(intensity=0.0, studio_light=0.7)
    out = apply(f, None, p, segmenter=FakeSeg())  # type: ignore[arg-type]
    assert out.shape == f.shape


# ---------------------------------------------------------------- real segmenter (cached-model)


def _model_cached() -> bool:
    from bedhead.models import MODEL_DIR

    return (MODEL_DIR / "selfie_multiclass_256x256.tflite").exists()


@pytest.mark.skipif(not _model_cached(), reason="segmenter model not cached yet")
def test_real_segmenter_masks_on_face_photo() -> None:
    from bedhead.segmenter import Segmenter

    img = cv2.imread("/tmp/face_lena.jpg")
    if img is None:
        pytest.skip("face fixture not present")
    canvas = np.zeros((720, 1280, 3), np.uint8)
    big = cv2.resize(img, (720, 720))
    canvas[:, 280:1000] = big

    seg = Segmenter()
    try:
        seg.tick(canvas, 0)
        seg.tick(canvas, 33)  # second tick exercises hold path
        pm = seg.person_mask()
        assert pm.shape == (720, 1280)
        # person should occupy the center band, not the edges
        assert pm[360, 640] > 0.8, "center should be person"
        assert pm[100, 30] < 0.2, "corner should be background"
    finally:
        seg.close()
