"""Identity-guard tests.

Cosine/admit/drift logic is tested with synthetic embeddings (no model
needed). A real-model test (marked `guard_model`) runs when insightface is
installed and its buffalo_l model is cached; it verifies admission behavior
on the calibration fixtures (same-person photos admitted, cross-person
rejected) using faces fetched to /tmp.
"""

from __future__ import annotations

import numpy as np
import pytest

from bedhead.guard import (
    ADMISSION_THRESHOLD,
    DRIFT_CAP,
    AdmissionResult,
    IdentityGuard,
    cosine,
)


def _rand_unit(rng, d=512):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- cosine


def test_cosine_identical() -> None:
    v = _rand_unit(np.random.default_rng(0))
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal() -> None:
    rng = np.random.default_rng(1)
    a = _rand_unit(rng)
    b = np.zeros_like(a)
    b[0], a[:2] = 1.0, 0.0
    a /= np.linalg.norm(a)
    assert abs(cosine(a, b)) < 1e-6


def test_cosine_unnormalized_inputs() -> None:
    a = np.array([3.0, 0.0])
    b = np.array([10.0, 0.0])
    assert cosine(a, b) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- admit (synthetic)


def test_admit_same_person() -> None:
    rng = np.random.default_rng(2)
    ref = _rand_unit(rng)
    # live samples = same identity + small noise
    live = [np.clip(ref + rng.normal(0, 0.05, 512), -1, 1) for _ in range(10)]
    g = object.__new__(IdentityGuard)  # admit() is pure math; no model needed
    r = g.admit(ref, live)
    assert isinstance(r, AdmissionResult)
    assert r.admitted, r.reason
    # different person rejected
    other = [_rand_unit(rng) for _ in range(10)]
    r2 = g.admit(ref, other)
    assert not r2.admitted


def test_admit_threshold_boundary() -> None:
    # construct embeddings at an exact cosine via shared direction + rotation
    rng = np.random.default_rng(3)
    ref = _rand_unit(rng)
    for target in (ADMISSION_THRESHOLD - 0.05, ADMISSION_THRESHOLD, ADMISSION_THRESHOLD + 0.05):
        # embed at angle with cos = target: e = target*ref + sqrt(1-t^2)*orth
        orth = _rand_unit(rng)
        orth -= ref * np.dot(ref, orth)
        orth /= np.linalg.norm(orth)
        e = target * ref + np.sqrt(max(0.0, 1 - target * target)) * orth
        assert cosine(ref, e) == pytest.approx(target, abs=1e-6)


def test_drift_cap_semantics() -> None:
    rng = np.random.default_rng(4)
    ref = _rand_unit(rng)
    # close render passes
    close = ref + rng.normal(0, 0.1, 512)
    close /= np.linalg.norm(close)
    assert cosine(ref, close) > DRIFT_CAP
    # different person fails
    other = _rand_unit(rng)
    assert cosine(ref, other) < DRIFT_CAP


def test_admission_result_shape() -> None:
    r = AdmissionResult(admitted=False, similarity=0.1, reason="x")
    assert r.admitted is False and r.similarity == 0.1


# ---------------------------------------------------------------- real model (optional)


def _guard_available() -> bool:
    try:
        import insightface  # noqa: F401

        return True
    except ImportError:
        return False


def _fixtures() -> dict[str, str]:
    import os

    paths = {
        "obama": "/tmp/bh_faces/obama.jpg",
        "obama2": "/tmp/bh_faces/obama2.jpg",
        "biden": "/tmp/bh_faces/biden.jpg",
    }
    return {k: v for k, v in paths.items() if os.path.exists(v)}


@pytest.mark.guard_model
@pytest.mark.skipif(not _guard_available() or len(_fixtures()) < 3,
                    reason="insightface or fixtures not available")
def test_real_admission_same_vs_cross_person() -> None:
    import cv2

    fx = _fixtures()
    g = IdentityGuard()
    obama = g.embed(cv2.imread(fx["obama"]))
    obama2 = g.embed(cv2.imread(fx["obama2"]))
    biden = g.embed(cv2.imread(fx["biden"]))
    assert obama is not None and obama2 is not None and biden is not None

    same = g.admit(obama, [obama2])
    assert same.admitted, f"same-person reference must be admitted ({same.reason})"
    cross = g.admit(biden, [obama, obama2])
    assert not cross.admitted, "cross-person reference must be rejected"
    assert same.similarity > 0.6
    assert cross.similarity < 0.2
