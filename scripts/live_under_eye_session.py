"""Live under-eye session: raw + retouched capture, benchmark metric, JSON out.

Run from a real Terminal (so the macOS camera prompt can appear):
  .venv/bin/python scripts/live_under_eye_session.py [seconds=8]

Grabs the camera at 720p, runs the shipped pipeline, measures the
benchmark's under_eye_gap (cheek p60 - band p10) on the last settled
frames both raw and processed, prints a table, writes JSON to
docs/live-under-eye-<date>.json for RESULTS.md.
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

from bedhead.config import Preset  # noqa: E402
from bedhead.retoucher import apply, under_eye_mask  # noqa: E402
from bedhead.tracker import FACE_OVAL, FaceTracker  # noqa: E402

W, H = 1280, 720


def band_gap(lab_l: np.ndarray, band: np.ndarray, oval: np.ndarray) -> float:
    cheek = (oval > 128) & ~band
    if band.sum() < 50 or cheek.sum() < 50:
        return float("nan")
    return float(np.percentile(lab_l[cheek], 60) - np.percentile(lab_l[band], 10))


def main(seconds: float = 8.0) -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("!! camera not available")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    tracker = FaceTracker()
    preset = Preset(intensity=0.85, under_eye=0.8, skin=0.6, shine=0.5)

    raw_gaps, out_gaps = [], []
    t_end = time.time() + seconds
    frame_i, missed = 0, 0
    try:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                missed += 1
                time.sleep(0.05)
                continue
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
            face = tracker.detect(frame, frame_i * 33)
            frame_i += 1
            if face is None:
                continue
            out = apply(frame, face, preset)
            band = under_eye_mask(face) > 128
            if band.sum() < 50:
                continue
            oval = np.zeros((H, W), np.uint8)
            cv2.fillPoly(oval, [face.poly(FACE_OVAL)], 255)
            g_raw = band_gap(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0], band, oval)
            g_out = band_gap(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0], band, oval)
            if np.isfinite(g_raw) and np.isfinite(g_out):
                raw_gaps.append(g_raw)
                out_gaps.append(g_out)
            if frame_i % 30 == 0:
                print(f"...{time.time():.0f} frames {frame_i} faces-metric {len(raw_gaps)}")
    finally:
        cap.release()
        tracker.close()

    if len(raw_gaps) < 5:
        print(f"!! only {len(raw_gaps)} measurable frames (face lock too short)")
        return 2

    # median of the last 5 settled samples (benchmark methodology)
    r = float(np.median(raw_gaps[-5:]))
    o = float(np.median(out_gaps[-5:]))
    print(f"\nframes captured: {frame_i} | metric frames: {len(raw_gaps)} | read-miss: {missed}")
    print(f"live under_eye_gap: raw {r:.1f} -> processed {o:.1f}  (target < 10) "
          f"{'PASS' if o < 10 else 'FAIL'}")
    dest = Path(__file__).resolve().parent.parent / "docs" / f"live-under-eye-{date.today()}.json"
    dest.write_text(json.dumps({
        "date": str(date.today()),
        "raw_median_last5": r,
        "processed_median_last5": o,
        "n_metric_frames": len(raw_gaps),
        "frames": frame_i,
        "preset": {"intensity": preset.intensity, "under_eye": preset.under_eye,
                   "skin": preset.skin, "shine": preset.shine},
    }, indent=2))
    print(f"saved: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 8.0))
