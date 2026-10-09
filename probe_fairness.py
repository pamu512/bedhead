"""Demographic fairness probe for bedhead's guard + pipeline.

For every downloaded photo (grouped East Asian / South Asian / African):
  1. ArcFace embedding (guard.embed) -> same-person & cross-person similarity
     distributions per group, vs ADMISSION_THRESHOLD (0.40)
  2. MediaPipe tracker detection success + landmark count
  3. Segmenter face-skin coverage inside face oval
  4. Full-pipeline apply() sanity (dtype/range, changed-pixel fraction)

The admission gate is fair if, per group, min(same-person) comfortably
exceeds the threshold and max(cross-person) stays far below it.
"""
import json
import itertools
from pathlib import Path

import cv2
import numpy as np

from bedhead.config import Preset
from bedhead.guard import ADMISSION_THRESHOLD, IdentityGuard
from bedhead.retoucher import apply
from bedhead.segmenter import Segmenter
from bedhead.tracker import FaceTracker, FACE_OVAL

ROOT = Path("/tmp/bh_fairness")
manifest = json.loads((ROOT / "manifest.json").read_text())

guard = IdentityGuard()
tracker = FaceTracker()
seg = Segmenter()
preset = Preset()  # defaults

emb: dict[str, np.ndarray] = {}
track_ok: dict[str, bool] = {}
seg_cov: list[float] = []
pipe = {"n": 0, "detect_fail": 0, "seg_fail": 0, "changed_frac": []}
_ts = [0]  # MediaPipe VIDEO mode: global strictly-increasing timestamp


def next_ts() -> int:
    _ts[0] += 33
    return _ts[0]


for key, entries in manifest.items():
    if not entries:
        continue
    for e in entries:
        p = e["file"]
        img = cv2.imread(p)
        if img is None:
            continue
        # guard embedding
        g = guard.embed(img)
        if g is not None:
            emb[p] = g
        # tracker
        ff = tracker.detect(img, next_ts())
        ok = ff is not None
        track_ok[p] = ok
        if not ok:
            pipe["detect_fail"] += 1
            continue
        # segmenter coverage inside oval
        seg.tick(img, next_ts())
        fm = seg.face_skin_mask()
        h, w = img.shape[:2]
        if h != fm.shape[0] or w != fm.shape[1]:
            fm2 = cv2.resize(fm, (w, h))
        else:
            fm2 = fm
        oval = ff.landmarks[list(FACE_OVAL), :2].astype(int)  # type: ignore[union-attr]
        x0, y0 = oval.min(axis=0); x1, y1 = oval.max(axis=0)
        cov = float(fm2[y0:y1, x0:x1].mean())
        seg_cov.append(cov)
        # pipeline
        out = apply(img, ff, preset)
        assert out.dtype == np.uint8 and out.shape == img.shape
        changed = float((np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 6).mean())
        pipe["changed_frac"].append(changed)
        pipe["n"] += 1

print(f"photos processed: {pipe['n']} | embedding ok: {len(emb)} | tracker fails: {pipe['detect_fail']}")

# ---- per-group similarity stats
def group_of(p: str) -> str:
    return Path(p).parts[-3]

by_group: dict[str, dict[str, list[float]]] = {}
for grp in ("east_asian", "south_asian", "african"):
    same, cross = [], []
    people: dict[str, list[np.ndarray]] = {}
    for p, e in emb.items():
        if group_of(p) != grp:
            continue
        person = Path(p).parent.name
        people.setdefault(person, []).append(e)
    for person, es in people.items():
        for a, b in itertools.combinations(es, 2):
            same.append(float(np.dot(a, b)))
    plist = list(people.items())
    for (pa, ea), (pb, eb) in itertools.combinations(plist, 2):
        for a in ea:
            for b in eb:
                cross.append(float(np.dot(a, b)))
    by_group[grp] = {"same": same, "cross": cross}
    if same and cross:
        print(f"\n{grp:12s} people={len(people)} same-pairs={len(same)} cross-pairs={len(cross)}")
        print(f"  same : min={min(same):.3f} mean={np.mean(same):.3f}")
        print(f"  cross: max={max(cross):.3f} mean={np.mean(cross):.3f}")
        margin_lo = min(same) - ADMISSION_THRESHOLD
        margin_hi = ADMISSION_THRESHOLD - max(cross)
        print(f"  margin to threshold 0.40: same-minus-thr={margin_lo:+.3f} thr-minus-cross={margin_hi:+.3f}")

# ---- tracker/segmenter/pipeline per group
print("\ntracker detection by group:")
for grp in ("east_asian", "south_asian", "african"):
    oks = [v for p, v in track_ok.items() if group_of(p) == grp]
    if oks:
        print(f"  {grp:12s} {sum(oks)}/{len(oks)} detected")

ok_paths = [k for k in track_ok if track_ok[k]]
cov_by: dict[str, list[float]] = {}
for p, c in zip(ok_paths, seg_cov):
    cov_by.setdefault(group_of(p), []).append(c)
print("\nsegmenter face-skin coverage (inside oval):")
for grp, cs in cov_by.items():
    print(f"  {grp:12s} mean={np.mean(cs):.3f} min={np.min(cs):.3f}")

print(f"\npipeline: {pipe['n']} photos; changed-pixel fraction mean={np.mean(pipe['changed_frac']):.3f} "
      f"min={np.min(pipe['changed_frac']):.3f} max={np.max(pipe['changed_frac']):.3f}")

tracker.close()
seg.close()
print("FAIRNESS PROBE DONE")
