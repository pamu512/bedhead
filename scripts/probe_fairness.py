"""Fairness probe v3: video-call framing, single-face filter, per-group stats.

Protocol per photo:
  1. insightface must find exactly ONE face (crowd photos excluded)
  2. normalize to video-call framing: crop 3.5x face width, 16:9, 1280x720
     (a webcam user's face occupies a comparable fraction of the frame)
  3. run the REAL pipeline components on the normalized frame:
     tracker.detect -> segmenter.tick -> apply()
Measure per group (east_asian / south_asian / african):
  - ArcFace same-person pair similarity (gallery vs "live" use case)
  - ArcFace cross-person max (worst-case false-admit)
  - tracker lock rate, segmenter face-skin coverage, pipeline delta
"""
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
W, H = 1280, 720

guard = IdentityGuard()
tracker = FaceTracker()
seg = Segmenter()
preset = Preset()


def normalize(img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Single-face check + video-call framing crop. Returns (frame, embedding)."""
    dets = guard._app.get(img)
    if len(dets) != 1:
        return None, None
    x0, y0, x1, y1 = dets[0].bbox
    fw = max((x1 - x0) * 3.5, 64)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - (y1 - y0) * 0.35  # headroom like a webcam
    h, w = img.shape[:2]
    cw = min(int(fw), w)
    ch = min(int(cw * 9 / 16), h)
    if cw * 9 > 16 * ch:
        cw = int(ch * 16 / 9)
    px = int(np.clip(cx - cw / 2, 0, w - cw))
    py = int(np.clip(cy - ch / 2, 0, h - ch))
    crop = cv2.resize(img[py:py + ch, px:px + cw], (W, H))
    return crop, dets[0].normed_embedding


data: dict[str, dict[str, list[dict]]] = {}  # group -> person -> [frames]
ts = 0
stats = {"tracker": {}, "seg_cov": {}, "delta": {}}

for f in sorted(ROOT.glob("*/*/*.jpg")) + sorted(ROOT.glob("*/*/*.png")):
    img = cv2.imread(str(f))
    if img is None:
        continue
    frame, emb = normalize(img)
    if frame is None or emb is None:
        continue
    group, person = f.parts[-3], f.parts[-2]
    ts += 33
    ff = tracker.detect(frame, ts)
    entry = {"emb": emb, "frame": frame, "tracked": ff is not None}
    if ff is not None:
        oval = ff.landmarks[list(FACE_OVAL), :2].astype(int)
        seg.tick(frame, ts, face_oval_pts=oval)
        fm = seg.face_skin_mask()
        oval = ff.landmarks[list(FACE_OVAL), :2].astype(int)
        x0, y0 = oval.min(axis=0); x1, y1 = oval.max(axis=0)
        cov = float(fm[y0:y1, x0:x1].mean())
        out = apply(frame, ff, preset)
        delta = float((np.abs(out.astype(int) - frame.astype(int)).sum(axis=2) > 6).mean())
        entry["seg_cov"] = cov
        entry["delta"] = delta
        stats["seg_cov"].setdefault(group, []).append(cov)
        stats["delta"].setdefault(group, []).append(delta)
    stats["tracker"].setdefault(group, []).append(ff is not None)
    data.setdefault(group, {}).setdefault(person, []).append(entry)

print("=== usable single-face photos (video-call framing) ===")
for grp, people in data.items():
    n = sum(len(v) for v in people.values())
    print(f"  {grp:12s} {len(people)} people, {n} photos, "
          f"per-person={[len(v) for v in people.values()]}")

print("\n=== ArcFace similarity vs threshold 0.40 ===")
verdicts = []
for grp, people in data.items():
    same, cross = [], []
    for es in people.values():
        embs = [e["emb"] for e in es if e["tracked"]]
        for a, b in combinations(embs, 2):
            same.append(float(np.dot(a, b)))
    persons = list(people.items())
    for (pa, ea), (pb, eb) in combinations(persons, 2):
        for a in ea:
            for b in eb:
                cross.append(float(np.dot(a["emb"], b["emb"])))
    print(f"\n{grp}:")
    if same:
        print(f"  same-person : n={len(same):3d} min={min(same):.3f} p25={np.percentile(same,25):.3f} "
              f"mean={np.mean(same):.3f} max={max(same):.3f}")
    else:
        print("  same-person : (no person with >=2 tracked photos)")
    if cross:
        print(f"  cross-person: n={len(cross):3d} max={max(cross):.3f} mean={np.mean(cross):.3f}")
    same_ok = same and min(same) >= ADMISSION_THRESHOLD
    cross_ok = cross and max(cross) < ADMISSION_THRESHOLD
    verdicts.append((grp, same_ok, cross_ok, min(same) if same else None,
                     max(cross) if cross else None))
    print(f"  admission: same>=0.40 {'PASS' if same_ok else ('FAIL' if same else 'n/a')}"
          f" | cross<0.40 {'PASS' if cross_ok else ('FAIL' if cross else 'n/a')}")

print("\n=== tracker lock rate (video framing) ===")
for grp, oks in stats["tracker"].items():
    print(f"  {grp:12s} {sum(oks)}/{len(oks)} = {100*np.mean(oks):.0f}%")

print("\n=== segmenter face-skin coverage in oval ===")
for grp, cs in stats["seg_cov"].items():
    print(f"  {grp:12s} n={len(cs)} mean={np.mean(cs):.3f} min={np.min(cs):.3f}")

print("\n=== pipeline changed-pixel fraction ===")
for grp, cs in stats["delta"].items():
    print(f"  {grp:12s} n={len(cs)} mean={np.mean(cs):.4f} min={np.min(cs):.4f} max={np.max(cs):.4f}")

tracker.close()
seg.close()

worst_same = min((v[3] for v in verdicts if v[3] is not None), default=None)
best_cross = max((v[4] for v in verdicts if v[4] is not None), default=None)
print(f"\nWORST same-person similarity across groups: {worst_same:.3f}" if worst_same else "")
print(f"BEST cross-person similarity across groups: {best_cross:.3f}" if best_cross else "")
print("FAIRNESS v3 DONE")
