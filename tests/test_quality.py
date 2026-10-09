"""Tests for the quality primitives (one-euro, schmitt, guided separation)."""
from __future__ import annotations

import numpy as np

from bedhead.quality import OneEuro, Schmitt, frequency_separation


def test_one_euro_kills_jitter():
    rng = np.random.default_rng(7)
    oe = OneEuro(1, min_cutoff=1.2, beta=0.02)
    still = np.array([[50.0, 60.0]], np.float32)
    jittered = still + rng.normal(0, 0.6, (300, 2)).astype(np.float32)
    filtered = np.array([oe(p) for p in jittered])
    assert np.abs(filtered - still).mean() < np.abs(jittered - still).mean()
    fast = np.array([[100.0, 100.0]], np.float32)
    out = oe(fast)
    assert abs(out[0, 0] - 100.0) > 10  # tracks fast motion with low lag


def test_schmitt_hysteresis():
    s = Schmitt(hi=0.30, lo=0.18)
    assert not s(0.29)
    assert s(0.31)      # crosses hi -> on
    assert s(0.25)      # stays on inside the band (no flicker)
    assert not s(0.17)  # crosses lo -> off


def test_guided_separation_preserves_edges():
    rng = np.random.default_rng(1)
    img = np.full((120, 120, 3), 128, np.uint8)
    img[:, 60:] = 200
    img = (img.astype(np.float32) + rng.normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    base, detail = frequency_separation(img)
    step = base[:, 65].mean() - base[:, 55].mean()
    assert step > 40          # edge survives in the base layer
    assert np.abs(detail).mean() < 15  # detail carries only noise
