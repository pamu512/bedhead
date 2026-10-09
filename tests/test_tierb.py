"""Tier B guard-contract tests (no heavy model needed: swapper monkeypatched)."""
from __future__ import annotations

import numpy as np
import pytest

from bedhead.tierb import DRIFT_CHECK_INTERVAL, TierB, TierBError


def _unit(d=512, seed=0):
    v = np.random.default_rng(seed).normal(size=d)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeFace:
    def __init__(self, emb=None, kps=None):
        self.bbox = np.array([10, 10, 100, 100], np.float32)
        # sane 5-point kps (eyes, nose, mouth corners) for alignment geometry
        self.kps = kps if kps is not None else np.array(
            [[38, 48], [90, 48], [64, 72], [44, 96], [84, 96]], np.float32)
        self.normed_embedding = emb if emb is not None else _unit(3)
        self.det_score = 0.9


class FakeGuard:
    """Minimal IdentityGuard stand-in keyed by image object id.

    `embs` maps id -> embedding for registered arrays; `default_emb` is
    returned for anything else (copies of registered arrays included).
    """

    def __init__(self, default_emb=None):
        self.faces: dict[int, list] = {}
        self.embs: dict[int, np.ndarray] = {}
        self.default_emb = default_emb

    def embed(self, img):
        return self.embs.get(id(img), self.default_emb)

    @property
    def _app(self):
        guard = self

        class App:
            def get(self, img):
                return guard.faces.get(id(img), [])

        return App()


def _bare_tierb(guard) -> TierB:
    tb = object.__new__(TierB)
    tb.guard = guard
    tb._swapper = None
    tb._reference_face = None
    tb._reference_embedding = None
    tb._disabled_reason = None
    tb._frames_since_check = 0
    tb._drift_strikes = 0
    tb._blend_scale = 1.0
    tb._last_drift_sim = 1.0
    tb.last_swap_ms = 0.0
    tb._model_path = ""
    return tb


@pytest.fixture()
def setup():
    rng = np.random.default_rng(1)
    ref_img = (rng.normal(0, 5, (64, 64, 3)) + 120).clip(0, 255).astype(np.uint8)
    ref_emb = _unit(seed=7)
    guard = FakeGuard()
    guard.faces[id(ref_img)] = [FakeFace(emb=ref_emb)]
    guard.embs[id(ref_img)] = ref_emb
    return ref_img, ref_emb, guard


def test_set_reference_requires_a_face():
    tb = _bare_tierb(FakeGuard())
    rng = np.random.default_rng(2)
    img = rng.normal(0, 5, (32, 32, 3)).astype(np.uint8)
    with pytest.raises(TierBError):
        tb.set_reference(img)


def test_process_refuses_before_reference_registered():
    tb = _bare_tierb(FakeGuard())
    frame = np.zeros((32, 32, 3), np.uint8)
    out, status = tb.process(frame, frame, blend=1.0)
    assert status == "passthrough" and out is frame


def test_drift_cap_disables_tier_b(setup):
    ref_img, _ref_emb, guard = setup
    guard.default_emb = _unit(seed=99)  # unknown imgs (incl. copies): orthogonal
    tb = _bare_tierb(guard)
    tb.set_reference(ref_img)
    assert tb.reference_ready

    live = np.zeros((32, 32, 3), np.uint8)
    guard.faces[id(live)] = [FakeFace(emb=_unit(seed=8))]

    swapped = np.ones((32, 32, 3), np.uint8) * 200
    guard.embs[id(swapped)] = _unit(seed=99)  # orthogonal to ref_emb (seed 7)

    class SwapperReturnsProto:
        def get(self, frame, face, ref, paste_back=True):
            return swapped.copy()

    tb._swapper = SwapperReturnsProto()

    for _ in range(DRIFT_CHECK_INTERVAL + 1):
        _out, _st = tb.process(live, live, blend=1.0)
    # progressive fail-safe: first strike cut the blend (blend_scale 0.5);
    # per-frame status stays "ok" because the reduced blend still renders
    assert tb._drift_strikes >= 1 and tb._blend_scale < 1.0
    # keep violating: three strikes disable Tier B for the run
    for _ in range(DRIFT_CHECK_INTERVAL * 2 + 1):
        _o, status2 = tb.process(live, live, blend=1.0)
    assert status2.startswith("disabled"), status2
    assert tb.disabled_reason is not None and "drift" in tb.disabled_reason
    out3, status3 = tb.process(live, live, blend=1.0)
    assert status3.startswith("disabled") and out3 is live


def test_good_identity_keeps_running(setup):
    ref_img, ref_emb, guard = setup
    guard.default_emb = ref_emb  # copies of swapped report the SAME identity
    tb = _bare_tierb(guard)
    tb.set_reference(ref_img)

    live = np.zeros((32, 32, 3), np.uint8)
    guard.faces[id(live)] = [FakeFace(emb=ref_emb)]

    swapped = np.ones((32, 32, 3), np.uint8) * 200
    guard.embs[id(swapped)] = ref_emb  # identical identity: drift ~1.0

    class SwapperReturnsProto:
        def get(self, frame, face, ref, paste_back=True):
            return swapped.copy()

    tb._swapper = SwapperReturnsProto()
    for _ in range(DRIFT_CHECK_INTERVAL * 3):
        out, status = tb.process(live, live, blend=1.0)
    assert status == "ok"
    assert tb.disabled_reason is None
    assert out is not live  # swap applied


def test_blend_zero_is_tier_a_passthrough(setup):
    ref_img, _ref_emb, guard = setup
    tb = _bare_tierb(guard)
    tb.set_reference(ref_img)
    frame = np.zeros((32, 32, 3), np.uint8)
    out, status = tb.process(frame, frame, blend=0.0)
    assert status == "passthrough" and out is frame


def test_blend_mixes_frames(setup):
    ref_img, ref_emb, guard = setup
    tb = _bare_tierb(guard)
    tb.set_reference(ref_img)
    live = np.zeros((128, 128, 3), np.uint8)
    guard.faces[id(live)] = [FakeFace(emb=ref_emb)]
    swapped = np.full((128, 128, 3), 200, np.uint8)
    guard.embs[id(swapped)] = ref_emb

    class SwapperReturnsProto:
        def get(self, frame, face, ref, paste_back=True):
            return swapped.copy()

    tb._swapper = SwapperReturnsProto()
    tier_a = np.full((128, 128, 3), 100, np.uint8)
    out, status = tb.process(live, tier_a, blend=0.5)
    assert status == "ok"
    # blend scales the LOW-frequency swap contribution only (no RGB mixing,
    # which would cancel high-frequency phase): flat frame, raw=0, swap=200,
    # blend 0.5 -> low = 0*(1-0.5) + 200*0.5 = 100 at the face center
    assert abs(int(out[64, 64, 0]) - 100) <= 2
    # full blend -> low = 200
    out_full, _ = tb.process(live, tier_a, blend=1.0)
    assert abs(int(out_full[64, 64, 0]) - 200) <= 2
