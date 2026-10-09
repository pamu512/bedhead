"""Headless tests for the clothes tidy-up module.

Perception quality (does the segmenter see a real garment?) is validated in
live camera runs, not here: the multiclass model is trained on real people and
reads synthetic cartoon torsos as background. What IS testable headless:
effect math against constructed masks, fail-safe behavior, and guard rails.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bedhead.clothes import ClothesSegmenter, _garment_median, _mask_bbox, tidy
from bedhead.config import Preset


def _shirt_frame(w: int = 640, h: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic frame plus a hand-built feathered garment mask over the torso."""
    f = np.full((h, w, 3), (40, 50, 60), np.uint8)  # background
    cv2.ellipse(f, (w // 2, 520), (240, 220), 0, 180, 360, (60, 90, 160), -1)  # shirt body
    for i in range(4):                                            # creases
        cv2.ellipse(f, (w // 2, 470 - i * 28), (200 - i * 30, 10), 0, 0, 180, (40, 65, 130), 2)
    cv2.circle(f, (w // 2 + 60, 430), 14, (30, 50, 110), -1)      # faint stain
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, 520), (238, 218), 0, 180, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return f, mask


@pytest.fixture(scope="module")
def segmenter():
    try:
        seg = ClothesSegmenter()
    except Exception as e:  # pragma: no cover  # noqa: BLE001
        pytest.skip(f"segmenter unavailable: {e}")
    yield seg
    seg.close()


# ---------------------------------------------------------------- tidy math

def test_tidy_output_finite_and_bounded():
    frame, mask = _shirt_frame()
    out = tidy(frame.copy(), mask, Preset(intensity=1.0, clothes=0.7, stain=0.6))
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float32)).all()
    # far background (outside mask + feather) must be untouched
    assert np.array_equal(out[:40, :40], frame[:40, :40])


def test_tidy_crease_energy_drops():
    frame, mask = _shirt_frame()
    before = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    out = tidy(frame.copy(), mask, Preset(intensity=1.0, clothes=0.8, stain=0.0))
    after = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert after < before, f"creases should soften ({before:.0f} -> {after:.0f})"


def test_tidy_stain_pulls_toward_median():
    frame, mask = _shirt_frame()
    out = tidy(frame.copy(), mask, Preset(intensity=1.0, clothes=0.0, stain=0.8))
    # the stain pixel neighborhood should move toward shirt color (B 60-ish)
    y, x = 430, 320 + 60
    before_dist = abs(int(frame[y, x, 0]) - 60)
    after_dist = abs(int(out[y, x, 0]) - 60)
    assert after_dist < before_dist, f"stain should fade ({before_dist} -> {after_dist})"


def test_tidy_none_mask_passthrough():
    frame, _ = _shirt_frame()
    out = tidy(frame, None, Preset(intensity=1.0))
    assert out is frame


def test_tidy_zero_strength_passthrough():
    frame, mask = _shirt_frame()
    out = tidy(frame.copy(), mask, Preset(intensity=0.0))
    assert np.array_equal(out, frame)


def test_tidy_garment_silhouette_survives():
    """Strong edges (shirt outline) must not be smeared into background."""
    frame, mask = _shirt_frame()
    out = tidy(frame.copy(), mask, Preset(intensity=1.0, clothes=1.0, stain=1.0, logo_blur=1.0))
    # sample a point just outside the shirt edge: should stay background
    assert tuple(out[10, 320]) == tuple(frame[10, 320])


# ---------------------------------------------------------------- logo blur

def _logo_shirt_frame(w: int = 640, h: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Shirt frame with a compact high-contrast 'logo' print on the chest."""
    f, mask = _shirt_frame(w, h)
    chest = (w // 2, 430)
    cv2.rectangle(f, (chest[0] - 30, chest[1] - 18), (chest[0] + 30, chest[1] + 18), (255, 255, 255), -1)
    cv2.putText(f, "LOGO", (chest[0] - 27, chest[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
    return f, mask


def test_logo_blur_flattens_print():
    """A compact high-contrast print must lose detail (variance drop) under blur."""
    frame, mask = _logo_shirt_frame()
    p = Preset(intensity=1.0, clothes=0.0, stain=0.0, logo_blur=1.0)
    out = tidy(frame.copy(), mask, p)
    y, x = 430, 320
    win_in = frame[y - 14:y + 14, x - 34:x + 34].astype(np.float32)
    win_out = out[y - 14:y + 14, x - 34:x + 34].astype(np.float32)
    assert win_out.var() < win_in.var() * 0.6, (
        f"print variance should drop sharply ({win_in.var():.0f} -> {win_out.var():.0f})"
    )


def test_logo_blur_off_leaves_print():
    """Same frame, dial off: print must survive untouched."""
    frame, mask = _logo_shirt_frame()
    out = tidy(frame.copy(), mask, Preset(intensity=1.0, clothes=0.0, stain=0.0, logo_blur=0.0))
    assert np.array_equal(out, frame)


def test_logo_blur_preserves_stripes():
    """Wide stripes (large-area deviation) are pattern, not prints: no redaction."""
    f = np.full((480, 640, 3), (40, 50, 60), np.uint8)
    for yy in range(250, 480, 44):  # horizontal stripes across the torso area
        f[yy:yy + 22, 100:540] = (200, 200, 200)
    mask = np.zeros((480, 640), np.uint8)
    mask[250:480, 100:540] = 255
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    out = tidy(f.copy(), mask, Preset(intensity=1.0, clothes=0.0, stain=0.0, logo_blur=1.0))
    # stripe edge sharpness must survive: gradient magnitude inside garment similar
    g_in = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    g_out = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert g_out > g_in * 0.85, f"stripes should survive ({g_in:.0f} -> {g_out:.0f})"


# ---------------------------------------------------------------- guard rails

def test_mask_bbox_rejects_slivers():
    m = np.zeros((100, 100), np.uint8)
    m[10:12, 10:40] = 255  # 2px tall sliver
    assert _mask_bbox(m) is None


def test_mask_bbox_bounds():
    m = np.zeros((100, 100), np.uint8)
    m[20:80, 20:80] = 255
    x0, y0, x1, y1 = _mask_bbox(m)
    assert x0 <= 20 and y0 <= 20 and x1 >= 79 and y1 >= 79
    assert 0 <= x0 < x1 < 100 and 0 <= y0 < y1 < 100


def test_garment_median_strided():
    roi = np.zeros((64, 64, 3), np.uint8)
    roi[:] = (50, 100, 150)
    m = np.full((64, 64), 255, np.uint8)
    med = _garment_median(roi, m)
    assert np.allclose(med, (50, 100, 150), atol=2)


# ---------------------------------------------------------------- segmenter

def test_segmenter_init_and_failsafe(segmenter):
    """Model loads, runs headless, and a no-garment frame yields None (fail-safe)."""
    frame = np.full((480, 640, 3), (40, 50, 60), np.uint8)  # empty room
    mask = segmenter.clothes_mask(frame, 0)
    # empty frame must NOT produce a garment mask
    assert mask is None
    # and tidy with None is a clean passthrough
    out = tidy(frame, mask, Preset(intensity=1.0))
    assert out is frame


def test_segmenter_never_crashes_on_noise(segmenter):
    """Random noise frames: mask or None, but never an exception."""
    rng = np.random.default_rng(3)
    for i in range(5):
        f = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = segmenter.clothes_mask(f, 10_000 + i * 33)
        assert mask is None or (mask.shape == f.shape[:2] and mask.dtype == np.uint8)
