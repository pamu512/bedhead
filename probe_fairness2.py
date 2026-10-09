"""Fairness probe v2: single-face photos only (largest face = the person).

Commons event photos contain crowds; a multi-face photo proves nothing about
the named person. Keep only photos where insightface detects exactly 1 face
(and MediaPipe tracker also locks on), then measure per group:
  - ArcFace same-person vs cross-person similarity vs ADMISSION_THRESHOLD
  - tracker detection, segmenter face-skin coverage, pipeline delta
"""
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from bedhead.config import Preset
from bedhead.guard import ADMISSION_THRESHOLD, IdentityGuard
from bedhead.retoucher import apply
from bedhead.segmenter import Segmenter
from bedhead.tracker import FACE_OVAL, FaceTracker

ROOT = Path("/tmp/bh_fairness")
manifest = json.loads((ROOT / "manifest.json").read_text())

guard = IdentityGuard()
tracker = FaceTracker()
seg = Segmenter()
preset = Preset()

emb: dict[str, np.ndarray] = {}
seg_cov: dict[str, list[float]] = {}
changed: dict[str, list[float]] = {}
ts = 0

for key, entries in manifest.items():
    if not entries:
        continue
    for e in entries:
        p = e["file"]
        img = cv2.imread(p)
        if img is None:
            continue
        n_faces = len(guard._app.get(img))
        if n_faces != 1:
            continue  # crowd/zero-face photo: not usable for this probe
        ts += 33
        ff = tracker.detect(img, ts)
        if ff is None:
            print(f"  [tracker-miss] {p}")
            continue
        g = guard.embed(img)
        assert g is not None
        emb[p] = g
        seg.tick(img, ts)
        fm = seg.face_skin_mask()
        h, w = img.shape[:2]
        if fm.shape != (h, w):
            fm = cv2.resize(fm, (w, h))
        oval = ff.landmarks[list(FACE_OVAL), :2].astype(int)
        x0, y0 = oval.min(axis=0); x1, y1 = oval.max(axis=0)
        grp = Path(p).parts[-3]
        seg_cov.setdefault(grp, []).append(float(fm[y0:y1, x0:x1].mean()))
        out = apply(img, ff, preset)
        changed.setdefault(grp, []).append(
            float((np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 6).mean())
        )

print(f"\nsingle-face photos used: {len(emb)}")


def group_of(p: str) -> str:
    return Path(p).parts[-3]


print("\n=== ArcFace similarity by group (single-face photos) ===")
for grp in ("east_asian", "south_asian", "african"):
    people: dict[str, list[np.ndarray]] = {}
    for p, e in emb.items():
        if group_of(p) == grp:
            people.setdefault(Path(p).parent.name, []).append(e)
    same, cross = [], []
    for es in people.values():
        for a, b in combinations(es, 2):
            same.append(float(np.dot(a, b)))
    for (pa, ea), (pb, eb) in combinations(list(people.items()), 2):
        for a in ea:
            for b in eb:
                cross.append(float(np.dot(a, b)))
    n_people = len(people)
    per_person = {k: len(v) for k, v in people.items()}
    print(f"\n{grp}: {n_people} people, photos/person={per_person}")
    if same:
        print(f"  same : n={len(same)} min={min(same):.3f} mean={np.mean(same):.3f}")
    if cross:
        print(f"  cross: n={len(cross)} max={max(cross):.3f} mean={np.mean(cross):.3f}")
    if same and cross:
        print(f"  threshold 0.40: worst same={min(same):.3f} -> "
              f"{'PASS (all same admitted)' if min(same) >= ADMISSION_THRESHOLD else 'FAIL: some same-person pairs below threshold'}")
        print(f"  best cross={max(cross):.3f} -> "
              f"{'PASS (all cross rejected)' if max(cross) < ADMISSION_THRESHOLD else 'FAIL: some cross-person pairs above threshold'}")

print("\n=== segmenter face-skin coverage inside oval ===")
for grp, cs in seg_cov.items():
    print(f"  {grp:12s} n={len(cs)} mean={np.mean(cs):.3f} min={np.min(cs):.3f} max={np.max(cs):.3f}")

print("\n=== pipeline changed-pixel fraction ===")
for grp, cs in changed.items():
    print(f"  {grp:12s} n={len(cs)} mean={np.mean(cs):.3f} min={np.min(cs):.3f} max={np.max(cs):.3f}")

tracker.close()
seg.close()
print("\nFAIRNESS PROBE v2 DONE")
