"""Reference-guided autotune tests (no models needed)."""
from __future__ import annotations

import cv2
import numpy as np

from bedhead.autotune import apply_autotune, autotune


def _img(L: float, b_chroma: float = 0.0, noise: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((120, 160, 3), L, np.float32)
    if b_chroma:
        base[..., 0] -= b_chroma * 0.5   # less blue
        base[..., 2] += b_chroma * 0.5   # more red
    base += rng.normal(0, noise, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def test_darker_live_suggests_soft_light():
    live = _img(90)          # dim office
    ref = _img(150)          # good-day photo
    r = autotune(live, ref)
    assert r.preset_delta["soft_light"] > 0.5
    assert any("darker" in n for n in r.notes)


def test_brighter_live_turns_soft_light_down():
    live = _img(170)
    ref = _img(120)
    r = autotune(live, ref)
    assert r.preset_delta["soft_light"] == 0.0


def test_warmer_reference_suggests_studio_light():
    live = _img(140, b_chroma=0)
    ref = _img(140, b_chroma=55)   # warm golden-hour look (b* gap ~16+)
    r = autotune(live, ref)
    assert r.preset_delta["studio_light"] > 0.5
    assert any("warmer" in n for n in r.notes)


def test_similar_looks_stay_conservative():
    live = _img(140, seed=1)
    ref = _img(142, seed=2)
    r = autotune(live, ref)
    assert r.preset_delta["soft_light"] <= 0.2
    assert r.preset_delta["studio_light"] == 0.0


def test_sharper_reference_reduces_skin_smoothing():
    rng = np.random.default_rng(3)
    base = np.full((240, 320, 3), 130, np.float32)
    texture = rng.normal(0, 12, (240, 320, 1)).repeat(3, axis=2)
    # reference = sharp texture, live = blurred version (webcam softness)
    ref = np.clip(base + texture, 0, 255).astype(np.uint8)
    live = cv2.GaussianBlur(ref, (0, 0), 3.0)
    r = autotune(live, ref)
    # blurrier live should NOT get extra smoothing; sharper ref lowers skin
    assert r.preset_delta["skin"] <= 0.65


def test_all_suggestions_clamped_0_1():
    live = _img(30)
    ref = _img(240, b_chroma=60, noise=8, seed=9)
    r = autotune(live, ref)
    for k, v in r.preset_delta.items():
        assert 0.0 <= v <= 1.0, (k, v)


def test_apply_autotune_merges_over_base():
    base = {"skin": 0.6, "intensity": 0.7, "background_mode": "blur"}
    r = autotune(_img(90), _img(160))
    merged = apply_autotune(base, r)
    assert merged["background_mode"] == "blur"      # untouched fields survive
    assert merged["skin"] == r.preset_delta["skin"]  # autotune wins
    assert set(merged) >= set(base)


def test_masked_stats_ignore_background():
    # bright background would skew unmasked stats; mask confines to a dim patch
    live = _img(150)
    live[40:80, 50:110] = 70
    mask = np.zeros((120, 160), np.float32)
    mask[40:80, 50:110] = 1.0
    ref = _img(150)
    r = autotune(live, ref, live_face_mask=mask)
    assert r.preset_delta["soft_light"] > 0.3  # sees the dim patch, not the bright field
