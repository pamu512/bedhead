"""Fairness re-measure: does gap-bridging recover the african-cohort lock gap?

Run:  python scripts/fairness_bridge_measure.py

Method (headless, fixtures under /tmp/bh_fairness):
  For every person with >= 3 usable photos, build a video-call-framed
  sequence from their photos (single-face normalization identical to
  tests/test_fairness.py). Then replay each sequence through FaceTracker
  twice:
    raw      - bridging disabled (bridge_max = 0): today's behavior
    bridged  - shipped default (bridge_max = 10)
  and count a frame "locked" when detect() returns a face. We also inject
  the documented failure correlate: a darkened copy of each frame (L* ~70,
  the african-cohort miss range) appended after the original, simulating
  the 1-3 frame flickers dark captures produce.

  Lock-rate = locked frames / total frames, per cohort, raw vs bridged.
  Bridging should never lower lock-rate (it only fills gaps), and should
  raise it exactly where raw detection flickers.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bedhead.tracker import FaceTracker  # noqa: E402

ROOT = Path("/tmp/bh_fairness")
W, H = 1280, 720
GROUPS = ("east_asian", "south_asian", "african")


def normalize(img: np.ndarray, face_det) -> np.ndarray | None:
    """Single-face video-call framing (same as tests/test_fairness.py)."""
    x0, y0, x1, y1 = face_det.bbox
    fw = max((x1 - x0) * 3.5, 64)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - (y1 - y0) * 0.35
    h, w = img.shape[:2]
    cw = min(int(fw), w)
    ch = min(int(cw * 9 / 16), h)
    if cw * 9 > 16 * ch:
        cw = int(ch * 16 / 9)
    px = int(np.clip(cx - cw / 2, 0, w - cw))
    py = int(np.clip(cy - ch / 2, 0, h - ch))
    return cv2.resize(img[py:py + ch, px:px + cw], (W, H))


def darken_to_l(frame: np.ndarray, target_l: float = 70.0) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur = float(np.median(lab[..., 0]))
    lab[..., 0] = np.clip(lab[..., 0] * (target_l / max(cur, 1.0)), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def build_sequences() -> dict[str, dict[str, list[np.ndarray]]]:
    """group -> person -> ordered frames (originals only, face-check via MP)."""
    from bedhead.tracker import FaceTracker as FT

    data: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    probe = FT()
    files = sorted(p for p in ROOT.glob("*/*/*") if p.suffix in (".jpg", ".png"))
    ts = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        ts += 33  # MediaPipe VIDEO mode demands monotonic timestamps
        small = cv2.resize(img, (640, 360))
        face = probe.detect(small, ts)
        if face is None:
            continue
        frame = normalize(img, _fake_det_from_landmarks(face, img.shape))
        if frame is not None:
            data[f.parts[-3]][f.parts[-2]].append(frame)
    probe.close()
    return data


class _FakeDet:
    def __init__(self, bbox):
        self.bbox = bbox


def _fake_det_from_landmarks(face, orig_shape: tuple):
    """bbox from mesh landmarks, mapped from probe space to original space."""
    pts = face.landmarks[:, :2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    oh, ow = orig_shape[:2]
    sx, sy = ow / face.w, oh / face.h
    return _FakeDet((x0 * sx, y0 * sy, x1 * sx, y1 * sy))


def run_sequence(tracker: FaceTracker, frames: list[np.ndarray]) -> tuple[int, int]:
    locked = 0
    for i, fr in enumerate(frames):
        f = tracker.detect(fr, i * 33)
        if f is not None:
            locked += 1
    return locked, len(frames)


def main() -> None:
    seqs = build_sequences()
    print(f"people: " + ", ".join(f"{g}={len(seqs.get(g, {}))}" for g in GROUPS))
    total_raw = total_brd = total_n = 0
    for group in GROUPS:
        people = seqs.get(group, {})
        if not people:
            continue
        raw_l = brd_l = n = 0
        for person, frames in people.items():
            if len(frames) < 2:
                continue
            # sequence: orig, dark(orig), orig, dark(orig), ... (flicker sim)
            seq = []
            for fr in frames:
                seq.append(fr)
                seq.append(darken_to_l(fr, 70.0))
            t_raw = FaceTracker()
            t_raw._bridge_max = 0            # noqa: SLF001 - measurement switch
            t_raw._bridge_left = 0
            l0, n0 = run_sequence(t_raw, seq)
            t_raw.close()
            t_brd = FaceTracker()
            l1, n1 = run_sequence(t_brd, seq)
            t_brd.close()
            raw_l += l0; brd_l += l1; n += n0
        if n:
            print(f"{group:12s}: raw {raw_l/n:5.1%} -> bridged {brd_l/n:5.1%}  (n={n})")
            total_raw += raw_l; total_brd += brd_l; total_n += n
    if total_n:
        print(f"{'ALL':12s}: raw {total_raw/total_n:5.1%} -> bridged {total_brd/total_n:5.1%}  (n={total_n})")


if __name__ == "__main__":
    main()
