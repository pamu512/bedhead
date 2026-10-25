"""Full-stack latency profile: every shipped effect at 720p, per-stage ms.

Run:  python scripts/latency_profile.py [portrait.jpg]

Stages mirror the live loop order in cli.py: tracker.detect (with
gap-bridging idle), segmentation tick (every 3rd frame), retoucher.apply
with all effect groups enabled (face effects + eye light + face lift +
vibrance + detail), clothes tidy (every 3rd frame), color match against
an admitted-reference stand-in. Reports per-stage mean/p95 and the
frame budget left at 30 fps (33.3 ms).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bedhead.clothes import ClothesSegmenter, tidy  # noqa: E402
from bedhead.config import Preset  # noqa: E402
from bedhead.retoucher import apply  # noqa: E402
from bedhead.segmenter import Segmenter  # noqa: E402
from bedhead.tracker import FaceTracker  # noqa: E402

N = 30  # measured frames (warm-up excluded)


def main(photo: str | None) -> None:
    img = None
    if photo:
        img = cv2.imread(photo)
    if img is None:
        default = Path(__file__).resolve().parent.parent / \
            "../one-piece-looks/apps/client/public/fixtures/face-portrait.jpg"
        img = cv2.imread(str(default)) if default.exists() else None
    if img is None:
        print("!! no portrait available (pass one as argv[1])")
        return
    frame = cv2.resize(img, (1280, 720))

    tracker = FaceTracker()
    seg = Segmenter()
    cseg = ClothesSegmenter()
    preset = Preset(
        intensity=0.85, skin=0.6, under_eye=0.8, shine=0.5, teeth=0.35,
        soft_light=0.25, vibrance=0.5, detail_boost=0.4, face_lift=0.3,
        eye_light=0.3, background_strength=0.6, background_mode="blur",
        studio_light=0.3, clothes=0.5, stain=0.4,
    )

    stages = {"track": [], "segment": [], "retouch": [], "clothes": [], "total": []}
    # warm-up (model load, caches)
    face = tracker.detect(frame, 0)
    seg.tick(frame, 0)
    out = apply(frame, face, preset, segmenter=seg)
    out = tidy(out, None, preset)

    for i in range(1, N + 1):
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        face = tracker.detect(frame, i * 33)
        t1 = time.perf_counter()
        seg.tick(frame, i * 33)          # internally every-3rd-frame amortized
        t2 = time.perf_counter()
        out = apply(frame, face, preset, segmenter=seg)
        t3 = time.perf_counter()
        cmask = cseg.clothes_mask_cached(frame, i * 33)
        out = tidy(out, cmask, preset)
        t4 = time.perf_counter()
        stages["track"].append((t1 - t0) * 1000)
        stages["segment"].append((t2 - t1) * 1000)
        stages["retouch"].append((t3 - t2) * 1000)
        stages["clothes"].append((t4 - t3) * 1000)
        stages["total"].append((t4 - t_total) * 1000)

    tracker.close(); seg.close(); cseg.close()

    print(f"full-stack latency, 720p, {N} frames (every effect on):")
    budget = 33.3
    for name, vals in stages.items():
        arr = np.array(vals)
        mark = "" if name == "total" else ""
        print(f"  {name:9s}: mean {arr.mean():5.1f} ms | p95 {np.percentile(arr, 95):5.1f} ms{mark}")
    t = np.array(stages["total"])
    fps = 1000.0 / t.mean()
    print(f"  -> end-to-end mean {t.mean():.1f} ms = {fps:.0f} fps ceiling "
          f"({(t.mean()/budget*100):.0f}% of a 30fps frame budget)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
