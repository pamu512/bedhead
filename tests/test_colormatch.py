"""Reference color match (Reinhard LAB transfer) tests."""
from __future__ import annotations

import numpy as np
import pytest

from bedhead.colormatch import MAX_CHROMA_SHIFT, color_match, lab_stats


def _img(L: float = 130.0, a_shift: float = 0.0, b_shift: float = 0.0,
         seed: int = 0) -> np.ndarray:

    rng = np.random.default_rng(seed)
    base = np.full((120, 160, 3), L, np.float32)
    base[..., 0] -= b_shift * 0.25   # blue down when warmer
    base[..., 2] += b_shift * 0.25   # red up when warmer
    base[..., 1] -= a_shift * 0.25
    base += rng.normal(0, 3, base.shape)
    img = np.clip(base, 0, 255).astype(np.uint8)
    # return BGR with the intended LAB stats via a quick pre-transform:
    return img


def test_zero_strength_is_passthrough():
    f = _img()
    assert color_match(f, _img(b_shift=30), 0.0) is f


def test_stats_match_at_full_strength():

    live = _img(L=100, seed=1)
    ref = _img(L=170, seed=2)
    out = color_match(live, ref, 1.0)
    lm, _ = lab_stats(live)
    om, _ = lab_stats(out)
    rm, _ = lab_stats(ref)
    # without masks the transfer is global: means must converge
    assert abs(om[0] - rm[0]) < 3.0
    assert abs(om[0] - lm[0]) > 30.0


def test_strength_scales_continuously():
    live = _img(L=100, seed=1)
    ref = _img(L=200, seed=2)
    lm, _ = lab_stats(live)
    rm, _ = lab_stats(ref)
    m50, _ = lab_stats(color_match(live, ref, 0.5))
    assert abs(m50[0] - lm[0]) < abs(rm[0] - lm[0])   # moved toward ref
    m100, _ = lab_stats(color_match(live, ref, 1.0))
    assert abs(m100[0] - rm[0]) < abs(m50[0] - rm[0])  # more at full strength


def test_mask_confines_transfer():

    rng = np.random.default_rng(5)
    live = np.full((100, 100, 3), 100, np.uint8) + rng.normal(0, 3, (100, 100, 3)).astype(int)
    live = np.clip(live, 0, 255).astype(np.uint8)
    ref = np.full((100, 100, 3), 220, np.uint8)
    mask = np.zeros((100, 100), np.float32)
    mask[20:80, 20:80] = 1.0
    out = color_match(live, ref, 1.0, face_mask=mask)
    # deep inside mask: brightened
    assert int(out[50, 50].sum()) > int(live[50, 50].sum()) + 100
    # far outside mask (corner): untouched (feather has ~12px reach)
    assert abs(int(out[2, 2].sum()) - int(live[2, 2].sum())) < 30


def test_chroma_shift_clamped():

    # live greenish vs extremely magenta reference: shift must clamp
    live = np.full((100, 100, 3), 130, np.uint8)
    ref = np.zeros((100, 100, 3), np.uint8)
    ref[..., 0] = 60    # low blue
    ref[..., 1] = 60    # low green
    ref[..., 2] = 255   # max red -> huge b* shift
    out = color_match(live, ref, 1.0)
    lm, _ = lab_stats(live)
    om, _ = lab_stats(out)
    # measured chroma movement never exceeds the clamp (+ small uint8 slack)
    assert abs(om[2] - lm[2]) <= MAX_CHROMA_SHIFT + 5.0


def test_identity_preserved_drift_cap():
    """ArcFace drift check on real faces: color transfer must not break identity."""
    import cv2

    from bedhead.guard import DRIFT_CAP, IdentityGuard

    ref = cv2.imread("/tmp/face_lena.jpg")
    if ref is None:
        pytest.skip("face fixture missing")
    # dim/cool live variant of the SAME face (as a webcam would see)
    live = np.clip(ref.astype(np.float32) * np.array([1.05, 0.85, 0.75]), 0, 255).astype(np.uint8)
    g = IdentityGuard()
    emb_ref = g.embed(ref)
    emb_matched = g.embed(color_match(live, ref, 1.0))
    assert emb_ref is not None and emb_matched is not None
    sim = float(np.dot(emb_ref, emb_matched))
    assert sim >= DRIFT_CAP, f"color match broke identity: {sim:.3f} < {DRIFT_CAP}"
