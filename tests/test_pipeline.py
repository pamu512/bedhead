"""Headless regression tests: no camera, no GUI, no virtual camera needed.

The retoucher is exercised through synthetic FaceFrame meshes (an ellipse
mesh is enough to drive every mask path); the tracker is only smoke-tested
for importability because creating a real FaceLandmarker requires the model
file (network on first run; cached afterwards).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bedhead.config import PRESETS, Preset
from bedhead.keys import key_from_code
from bedhead.retoucher import apply, hairline_band_mask, skin_mask, under_eye_mask
from bedhead.tracker import FaceFrame

W, H = 640, 360


def synth_mesh(w: int = W, h: int = H, seed: int = 0) -> FaceFrame:
    rng = np.random.default_rng(seed)
    lm = np.zeros((478, 3), np.float32)
    cx, cy, rx, ry = w / 2, h / 2, w * 0.20, h * 0.42
    for i in range(478):
        t = 2 * np.pi * i / 478
        lm[i, :2] = (cx + rx * np.cos(t), cy + ry * np.sin(t))
    lm[:, 0] = np.clip(lm[:, 0] + rng.normal(0, 2, 478), 5, w - 6)
    lm[:, 1] = np.clip(lm[:, 1] + rng.normal(0, 2, 478), 5, h - 6)
    return FaceFrame(landmarks=lm, h=h, w=w, extras={"jaw_open": 0.8})


def textured_frame(w: int = W, h: int = H) -> np.ndarray:
    frame = np.zeros((h, w, 3), np.uint8)
    for c in range(3):
        frame[:, :, c] = np.linspace(60 + 40 * c, 160 + 30 * c, w, dtype=np.uint8)[None, :]
    yy = np.repeat(np.linspace(0, 50, h, dtype=np.uint8)[:, None], w, axis=1)
    return cv2_blur(frame + yy[..., None])


def cv2_blur(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(img, (0, 0), 3)


# ---------------------------------------------------------------- keys


@pytest.mark.parametrize("code,expected", [
    (27, "\x1b"),       # Esc must quit (regression: used to be swallowed)
    (ord("q"), "q"),
    (ord(" "), " "),
    (ord("0"), "0"),
    (255, None),        # no-key sentinel
    (13, None),         # Enter is not printable ASCII; ignored
    (0, None),
])
def test_key_from_code(code: int, expected: str | None) -> None:
    assert key_from_code(code) == expected


# ---------------------------------------------------------------- presets


def test_preset_roundtrip(tmp_path) -> None:
    p = Preset(skin=0.3, intensity=0.9)
    p.save(str(tmp_path / "preset.json"))
    q = Preset.load(str(tmp_path / "preset.json"))
    assert q == p


def test_preset_rejects_non_numeric(tmp_path) -> None:
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"skin": "high"}))
    with pytest.raises(ValueError):
        Preset.load(str(f))


def test_preset_rejects_non_object(tmp_path) -> None:
    f = tmp_path / "bad.json"
    f.write_text("[1, 2, 3]")
    with pytest.raises((ValueError, TypeError)):
        Preset.load(str(f))


def test_preset_save_is_atomic_and_loadable_by_reader_midway(tmp_path) -> None:
    p = Preset()
    path = str(tmp_path / "preset.json")
    p.save(path)
    # Overwrite repeatedly; every intermediate read must parse (atomic replace).
    for i in range(20):
        p.intensity = i / 20
        p.save(path)
        Preset.load(path)  # must never raise


def test_named_presets_exist() -> None:
    assert set(PRESETS) >= {"subtle", "rescue", "studio", "focus"}
    # named presets must be loadable as Presets (values type-check)
    for overrides in PRESETS.values():
        p = Preset()
        for k, v in overrides.items():
            setattr(p, k, v)
        assert isinstance(p.describe(), str)


def test_scaled_clamps() -> None:
    p = Preset(intensity=2.0, skin=3.0)  # out-of-range values get clamped
    assert p.scaled("skin") == 1.0


# ---------------------------------------------------------------- retoucher


def test_no_face_passthrough_is_zero_copy() -> None:
    frame = textured_frame()
    out = apply(frame, None, Preset())
    assert out is frame


def test_intensity_zero_passthrough() -> None:
    frame = textured_frame()
    out = apply(frame, synth_mesh(), Preset(intensity=0.0))
    assert out is frame


def test_show_original_bypass() -> None:
    frame = textured_frame()
    out = apply(frame, synth_mesh(), Preset(show_original=True))
    assert out is frame


def test_all_effects_shape_and_range() -> None:
    frame = textured_frame()
    out = apply(frame, synth_mesh(), Preset())
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert 0 <= int(out.min()) and int(out.max()) <= 255


def _only(effect: str) -> Preset:
    """A preset with a single effect at full strength, everything else off."""
    p = Preset(intensity=1.0)
    for name in ("skin", "under_eye", "shine", "teeth", "hairline", "soft_light"):
        setattr(p, name, 1.0 if name == effect else 0.0)
    return p


def test_each_effect_isolated() -> None:
    frame = textured_frame()
    for name in ("skin", "under_eye", "shine", "teeth", "hairline", "soft_light"):
        out = apply(frame, synth_mesh(), _only(name))
        assert out.shape == frame.shape, name


def test_teeth_gate_dark_pixels_untouched() -> None:
    # Dark interior must not be brightened by the teeth effect.
    frame = textured_frame()
    face = synth_mesh()
    # Paint the mouth area dark (V near 0)
    frame[H // 2 - 10:H // 2 + 10, W // 2 - 30:W // 2 + 30] = 0
    out = apply(frame, face, _only("teeth"))
    dark_in = frame[H // 2 - 5:H // 2 + 5, W // 2 - 20:W // 2 + 20]
    dark_out = out[H // 2 - 5:H // 2 + 5, W // 2 - 20:W // 2 + 20]
    assert int(dark_out.max()) <= int(dark_in.max()) + 5, "dark mouth pixels must stay dark"


def test_masks_reference_frame_shapes() -> None:
    face = synth_mesh()
    assert skin_mask(face).shape == (H, W)
    assert under_eye_mask(face).shape == (H, W)
    assert hairline_band_mask(face).shape == (H, W)
    assert skin_mask(face).any(), "skin mask should cover some pixels"
    assert under_eye_mask(face).any()


def test_roi_confined_output_only_changes_face_region() -> None:
    # With soft_light off and only face effects on, pixels far outside the
    # face ROI must be bit-identical.
    frame = textured_frame()
    p = Preset(soft_light=0.0, intensity=1.0)
    out = apply(frame, synth_mesh(), p)
    corner = (slice(0, 20), slice(0, 20))
    assert np.array_equal(frame[corner], out[corner])
