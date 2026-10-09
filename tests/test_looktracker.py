"""Continuous auto-match (LookTracker) tests."""
from __future__ import annotations

import numpy as np

from bedhead.autotune import CONTINUOUS_FIELDS, LookTracker


def _img(L: float, b_chroma: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((120, 160, 3), L, np.float32)
    if b_chroma:
        base[..., 0] -= b_chroma * 0.5
        base[..., 2] += b_chroma * 0.5
    base += rng.normal(0, 2, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def test_prime_seeds_from_frame():
    lt = LookTracker(_img(150), interval_s=2.0)
    lt.prime(_img(90))  # dark live
    assert lt.current["soft_light"] > 0.5
    assert set(lt.current) == set(CONTINUOUS_FIELDS)


def test_tick_respects_interval():
    lt = LookTracker(_img(150), interval_s=2.0)
    lt.prime(_img(140))  # small gap: soft_light ~0.5 under the /20 mapping
    assert 0.3 < lt.current["soft_light"] < 0.8
    # first tick: clock sync only
    assert lt.tick(_img(140), now=100.0) is False
    # before the interval elapses: no re-measure
    assert lt.tick(_img(60), now=101.0) is False
    # after the interval: re-measures and adapts (frame got much darker ->
    # suggestion clamps to 1.0, current moves up)
    changed = lt.tick(_img(40), now=103.0)
    assert changed is True
    assert lt.current["soft_light"] > 0.6


def test_ema_smooths_toward_target():
    lt = LookTracker(_img(150), interval_s=1.0, ema=0.25)
    lt.prime(_img(100))               # soft_light suggestion ~1.0 from prime
    start = lt.current["soft_light"]
    lt.tick(_img(150), now=10.0)      # clock sync (no measure)
    # ambient brightens: target now 0; first measured tick moves 25% toward it
    assert lt.tick(_img(150), now=11.0) is True
    assert lt.current["soft_light"] < start
    assert lt.current["soft_light"] > start * 0.75 - 1e-6  # small step, not a jump
    # keep ticking: converges near 0
    for t in range(20):
        lt.tick(_img(150), now=20.0 + t * 2.0)
    assert lt.current["soft_light"] < 0.1


def test_deadband_suppresses_micro_changes():
    lt = LookTracker(_img(150), interval_s=1.0, deadband=0.03)
    lt.prime(_img(100))
    before = dict(lt.current)
    # a frame whose suggestion differs by less than the deadband
    lt.tick(_img(101), now=10.0)
    assert lt.current == before


def test_only_ambient_fields_touched():
    lt = LookTracker(_img(150, b_chroma=50), interval_s=1.0)
    lt.prime(_img(90))
    assert set(lt.current) == {"soft_light", "studio_light", "background_darken", "face_lift"}
    assert lt.current["studio_light"] > 0.0


def test_warm_room_adapts_studio_light_down():
    # reference warm; live starts neutral then the user warms the room lights
    lt = LookTracker(_img(140, b_chroma=50), interval_s=1.0)
    lt.prime(_img(140))                 # neutral live -> studio_light high
    hi = lt.current["studio_light"]
    assert hi > 0.3
    for t in range(12):
        lt.tick(_img(140, b_chroma=45), now=100.0 + t * 2.0)  # room warms up
    assert lt.current["studio_light"] < hi


def test_no_reference_flicker_on_steady_scene():
    # steady lighting: after settling, ticks report no change
    lt = LookTracker(_img(150), interval_s=1.0)
    lt.prime(_img(120))
    lt.tick(_img(120), now=10.0)       # settle
    steady = dict(lt.current)
    assert lt.tick(_img(120), now=12.0) is False
    assert lt.current == steady
