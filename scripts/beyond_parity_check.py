"""Beyond-parity check: under-eye v4 + tracker bridging, headless evidence.

Run:  python scripts/beyond_parity_check.py [path-to-portrait.jpg]

Measures the benchmark's own under_eye_gap metric (cheek p60 - band p10)
on a real photo through the live pipeline (FaceTracker -> retoucher.apply),
plus the synthetic-deficit sweep that pins the fix across the observed
raw range (46/64/82-level deficits). Prints a PASS/FAIL table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bedhead.config import Preset  # noqa: E402
from bedhead.retoucher import _retouch_roi, apply, under_eye_mask  # noqa: E402
from bedhead.tracker import FACE_OVAL, FaceTracker  # noqa: E402


def _gap(lab_l: np.ndarray, band: np.ndarray, oval: np.ndarray) -> float:
    cheek = (oval > 128) & ~band
    if band.sum() < 50 or cheek.sum() < 50:
        return float("nan")
    return float(np.percentile(lab_l[cheek], 60) - np.percentile(lab_l[band], 10))


def real_photo_check(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"!! could not read {path}")
        return
    frame = cv2.resize(img, (1280, 720))
    tracker = FaceTracker()
    face = tracker.detect(frame, 0)
    tracker.close()
    if face is None:
        print("!! no face detected in the photo; skipping real-photo check")
        return
    preset_rows = (
        ("default-ish", Preset(intensity=0.85, under_eye=0.8, skin=0.6, shine=0.5)),
        ("full dial  ", Preset(intensity=1.0, under_eye=1.0, skin=0.6, shine=0.5)),
    )
    band = under_eye_mask(face) > 128
    oval = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(oval, [face.poly(FACE_OVAL)], 255)
    raw_l = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
    raw = _gap(raw_l, band, oval)
    for label, preset in preset_rows:
        out = apply(frame, face, preset)
        fixed = _gap(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0], band, oval)
        verdict = "PASS" if fixed < 10.0 else "FAIL"
        print(f"real photo [{label}]: under_eye_gap raw {raw:5.1f} -> {fixed:5.1f}  (target < 10) {verdict}")


def synthetic_sweep() -> None:
    from tests.test_beyond_parity import _band_gap, _dark_trough_roi, _face

    face = _face()
    print("synthetic   : deficit  raw-gap -> fixed-gap  (target < 10)")
    for deficit in (40.0, 55.0, 70.0):
        roi = _dark_trough_roi(deficit)
        out = _retouch_roi(roi.copy(), face, Preset(intensity=1.0, under_eye=1.0))
        raw = _band_gap(roi, face)
        fixed = _band_gap(out, face)
        verdict = "PASS" if fixed < 10.0 else "FAIL"
        print(f"             {deficit:5.1f}  {raw:5.1f} -> {fixed:5.1f}  {verdict}")


if __name__ == "__main__":
    print("== bedhead beyond-parity check (under-eye v4 + tracker bridging) ==")
    synthetic_sweep()
    if len(sys.argv) > 1:
        real_photo_check(sys.argv[1])
    else:
        default = Path(__file__).resolve().parent.parent / "../one-piece-looks/apps/client/public/fixtures/face-portrait.jpg"
        if default.exists():
            real_photo_check(str(default))
        else:
            print("(pass a portrait path as argv[1] for the real-photo check)")
