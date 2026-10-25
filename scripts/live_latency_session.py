"""Live latency + temporal stability: the two reference-free bench metrics.

Run from a real Terminal (camera permission lives there):
  .venv/bin/python scripts/live_latency_session.py [seconds=10]

Measures per-frame end-to-end processing (tracker + segmenter + retoucher
with the FULL effect stack incl. blur/studio/vibrance + clothes tidy) at
720p live, plus frame-to-frame face-grade pumping (temporal metric).
Writes JSON next to the other live evidence docs.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bedhead.clothes import ClothesSegmenter, tidy
from bedhead.config import Preset
from bedhead.retoucher import apply
from bedhead.segmenter import Segmenter
from bedhead.tracker import FACE_OVAL, FaceTracker

W, H = 1280, 720


def main(seconds: float = 10.0) -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("!! camera not available")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    tracker = FaceTracker()
    seg = Segmenter()
    cseg = ClothesSegmenter()
    preset = Preset(
        intensity=0.85, skin=0.6, under_eye=0.8, shine=0.5, teeth=0.35,
        soft_light=0.25, vibrance=0.5, detail_boost=0.4, face_lift=0.3,
        eye_light=0.3, background_strength=0.6, background_mode="blur",
        studio_light=0.3, clothes=0.5, stain=0.4,
    )

    proc_ms: list[float] = []
    face_l_medians: list[float] = []
    t_end = time.time() + seconds
    frame_i = 0
    try:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
            t0 = time.perf_counter()
            face = tracker.detect(frame, frame_i * 33)
            seg.tick(frame, frame_i * 33)
            out = apply(frame, face, preset, segmenter=seg)
            cmask = cseg.clothes_mask_cached(frame, frame_i * 33)
            out = tidy(out, cmask, preset)
            proc_ms.append((time.perf_counter() - t0) * 1000)
            if face is not None:
                oval = np.zeros((H, W), np.uint8)
                cv2.fillPoly(oval, [face.poly(FACE_OVAL)], 255)
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0]
                sel = lab[oval > 128]
                if sel.size:
                    face_l_medians.append(float(np.median(sel)))
            frame_i += 1
    finally:
        cap.release()
        tracker.close()
        seg.close()
        cseg.close()

    arr = np.array(proc_ms[5:])  # drop warm-up
    lat_mean, lat_p95 = float(arr.mean()), float(np.percentile(arr, 95))
    # temporal: mean |dL| frame-to-frame on the face (grade pumping metric)
    if len(face_l_medians) > 10:
        deltas = np.abs(np.diff(face_l_medians))
        pump = float(deltas.mean())
    else:
        pump = float("nan")
    print(f"\nframes: {frame_i} | face-metric frames: {len(face_l_medians)}")
    print(f"live latency (full stack): mean {lat_mean:.1f} ms | p95 {lat_p95:.1f} ms"
          f"  (target < 45) {'PASS' if lat_mean < 45 else 'FAIL'}")
    print(f"temporal face-grade pumping: {pump:.2f} L*/frame  (target < 2.0) "
          f"{'PASS' if pump < 2.0 else 'FAIL'}")
    dest = Path(__file__).resolve().parent.parent / "docs" / f"live-latency-{date.today()}.json"
    dest.write_text(json.dumps({
        "date": str(date.today()),
        "latency_ms_mean": lat_mean,
        "latency_ms_p95": lat_p95,
        "temporal_pump_l_per_frame": pump,
        "frames": frame_i,
        "preset": "full stack (blur+studio+vibrance+clothes+face effects)",
    }, indent=2))
    print(f"saved: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 10.0))
