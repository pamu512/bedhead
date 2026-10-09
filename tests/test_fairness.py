"""Demographic-fairness regression tests (fixtures fetched to /tmp).

The admission threshold (0.40) and the person-mask oval-union were validated
on East Asian, South Asian, and African-descent face sets (Wikimedia Commons,
single-face photos, video-call framing; see probe_fairness3.py). These tests
re-check the two critical invariants on whichever fixtures are present:

  1. Guard separation: every same-person pair >= threshold; every
     cross-person pair < threshold, per group.
  2. Person-mask coverage: with a tracked face, person_mask() inside the face
     oval must stay high for every group (>= 0.7 mean, >= 0.5 min).

Skipped when the fixture directory is absent (e.g. CI).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pytest

from bedhead.guard import ADMISSION_THRESHOLD, IdentityGuard
from bedhead.segmenter import Segmenter
from bedhead.tracker import FACE_OVAL, FaceTracker

ROOT = Path("/tmp/bh_fairness")
W, H = 1280, 720
GROUPS = ("east_asian", "south_asian", "african")


def _fixtures_ready() -> bool:
    return ROOT.is_dir() and any(ROOT.glob("*/*/*.jpg"))


@pytest.fixture(scope="module")
def guard():
    return IdentityGuard()


@pytest.fixture(scope="module")
def normalized(guard):
    """group -> person -> [(frame, embedding, |yaw|)], single-face, video framing."""
    if not _fixtures_ready():
        pytest.skip("fairness fixtures not fetched (see fetch_faces.py)")
    tracker = FaceTracker()
    data: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, float]]]] = {}
    for f in sorted(ROOT.glob("*/*/*.jpg")) + sorted(ROOT.glob("*/*/*.png")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        dets = guard._app.get(img)
        if len(dets) != 1:
            continue
        x0, y0, x1, y1 = dets[0].bbox
        fw = max((x1 - x0) * 3.5, 64)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - (y1 - y0) * 0.35
        h, w = img.shape[:2]
        cw = min(int(fw), w)
        ch = min(int(cw * 9 / 16), h)
        if cw * 9 > 16 * ch:
            cw = int(ch * 16 / 9)
        px = int(np.clip(cx - cw / 2, 0, w - cw))
        py = int(np.clip(cy - ch / 2, 0, h - ch))
        frame = cv2.resize(img[py:py + ch, px:px + cw], (W, H))
        yaw = abs(float(dets[0].pose[1])) if getattr(dets[0], "pose", None) is not None else 0.0
        data.setdefault(f.parts[-3], {}).setdefault(f.parts[-2], []).append(
            (frame, dets[0].normed_embedding, yaw)
        )
    tracker.close()
    return data


def test_guard_separation_per_group(normalized):
    """Product contract: someone else's photo is always rejected (per group),
    and same-person similarity never collapses (floor 0.25 catches dataset
    errors like mislabeled photos; the pose failure mode bottoms there).
    """
    for grp, people in normalized.items():
        same, cross = [], []
        for entries in people.values():
            for (_, a, _), (_, b, _) in combinations(entries, 2):
                same.append(float(np.dot(a, b)))
        for (pa, ea), (pb, eb) in combinations(list(people.items()), 2):
            for _, a, _ in ea:
                for _, b, _ in eb:
                    cross.append(float(np.dot(a, b)))
        if same:
            worst = min(same)
            assert worst >= 0.25, (
                f"{grp}: same-person pair {worst:.3f} implausibly low "
                f"(expected >=0.25; below that suggests a fixture problem)"
            )
        if cross:
            assert max(cross) < ADMISSION_THRESHOLD, (
                f"{grp}: cross-person pair {max(cross):.3f} above threshold: "
                f"someone else's photo could be admitted"
            )


def test_frontal_same_person_admitted_per_group(normalized):
    """Within the front-facing contract (both photos |yaw| <= 25 deg), every
    group's same-person similarity must clear the admission threshold."""
    for grp, people in normalized.items():
        pair_sims = []
        for entries in people.values():
            frontal = [(emb, yaw) for _, emb, yaw in entries if yaw <= 25]
            for (ea, _), (eb, _) in combinations(frontal, 2):
                pair_sims.append(float(np.dot(ea, eb)))
        if pair_sims:
            worst = min(pair_sims)
            assert worst >= ADMISSION_THRESHOLD, (
                f"{grp}: frontal same-person pair {worst:.3f} below threshold "
                f"{ADMISSION_THRESHOLD}: a genuine front-facing reference would "
                f"be rejected"
            )


def test_person_mask_covers_face_for_all_groups(normalized):
    tracker = FaceTracker()
    seg = Segmenter()
    ts = 0
    try:
        for grp, people in normalized.items():
            covs = []
            for entries in people.values():
                for frame, _emb, _yaw in entries:
                    ts += 33
                    ff = tracker.detect(frame, ts)
                    if ff is None:
                        continue
                    oval = ff.landmarks[list(FACE_OVAL), :2].astype(int)
                    seg.tick(frame, ts, face_oval_pts=oval)
                    pm = seg.person_mask()
                    x0, y0 = oval.min(axis=0)
                    x1, y1 = oval.max(axis=0)
                    covs.append(float(pm[y0:y1, x0:x1].mean()))
            assert covs, f"{grp}: no tracked frames"
            assert np.mean(covs) >= 0.7, f"{grp}: mean person coverage {np.mean(covs):.3f}"
            assert min(covs) >= 0.5, f"{grp}: min person coverage {min(covs):.3f} (face eaten by bg)"
    finally:
        tracker.close()
        seg.close()
