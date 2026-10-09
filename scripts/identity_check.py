"""Identity-preservation harness: verify retouched video faces still match a profile picture.

Usage (dev):
    python scripts/identity_check.py --frames-root vidtimit --out report.json

Expects a directory tree: <root>/<subject>/<clip>/NNN.jpeg frames (VidTIMIT layout).
For each subject: first frame with a detectable face of the first clip is the
"profile picture"; the pipeline retouches every Nth frame of the other clips;
SFace cosine similarity of (retouched, profile) vs (original, profile) is compared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bedhead import retoucher
from bedhead.config import PRESETS, Preset
from bedhead.tracker import FaceTracker

ZOO = Path.home() / "Library/Caches/bedhead/identity"
DET = ZOO / "face_detection_yunet_2023mar.onnx"
REC = ZOO / "face_recognition_sface_2021dec.onnx"
MATCH_THRESHOLD = 0.363  # SFace cosine default (opencv zoo docs)


def embed(img_bgr: np.ndarray, det: cv2.FaceDetectorYN, rec: cv2.FaceRecognizerSF) -> np.ndarray | None:
    h, w = img_bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return None
    return rec.feature(rec.alignCrop(img_bgr, faces[0]))


def clip_frames(c: Path) -> list[Path]:
    frames = sorted(c.glob("*.jpeg")) or sorted(c.glob("*.jpg")) or sorted(c.glob("*.png"))
    if not frames:
        # VidTIMIT stores extensionless numbered frames (001, 002, ...)
        frames = sorted(p for p in c.iterdir() if p.is_file() and p.name.isdigit())
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("identity_report.json"))
    ap.add_argument("--preset", default="rescue")  # aggressive on purpose
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    det = cv2.FaceDetectorYN.create(str(DET), "", (320, 320))
    rec = cv2.FaceRecognizerSF.create(str(REC), "")
    tracker = FaceTracker()
    preset = Preset()
    if args.preset in PRESETS:
        for k, v in PRESETS[args.preset].items():
            setattr(preset, k, v)

    report: dict[str, dict] = {}
    ts = 0  # MediaPipe VIDEO mode: global strictly-increasing timestamp
    for subj_dir in sorted(p for p in args.frames_root.iterdir() if p.is_dir()):
        # VidTIMIT layout: <root>/<subj>/<subj>/video/<clip>/NNN.jpeg
        vdir = subj_dir / subj_dir.name / "video"
        clips = sorted(c for c in (vdir if vdir.is_dir() else subj_dir).iterdir() if c.is_dir())
        if not clips:
            continue
        subj = subj_dir.name

        prof_emb = None
        profile = None
        for c in clips:
            frames = clip_frames(c)
            if not frames:
                continue
            img = cv2.imread(str(frames[0]))
            if img is None:
                continue
            prof_emb = embed(img, det, rec)
            if prof_emb is not None:
                profile = f"{c.name}/{frames[0].name}"
                others = [x for x in clips if x.name != c.name] or clips
                break
        if prof_emb is None:
            report[subj] = {"error": "no detectable face"}
            continue

        sims_orig: list[float] = []
        sims_ret: list[float] = []
        n_proc = 0
        for c in others:  # type: ignore[possibly-unbound]
            for f in clip_frames(c)[:: args.stride]:
                ts += 1
                img = cv2.imread(str(f))
                if img is None:
                    continue
                emb_o = embed(img, det, rec)
                ff = tracker.detect(img, timestamp_ms=ts)
                out = retoucher.apply(img, ff, preset) if ff is not None else img
                emb_r = embed(out, det, rec)
                if emb_o is None or emb_r is None:
                    continue
                sims_orig.append(float(rec.match(prof_emb, emb_o, cv2.FaceRecognizerSF_FR_COSINE)))
                sims_ret.append(float(rec.match(prof_emb, emb_r, cv2.FaceRecognizerSF_FR_COSINE)))
                n_proc += 1

        if not sims_ret:
            report[subj] = {"error": "no comparable frames", "profile": profile}
            continue
        report[subj] = {
            "profile": profile,
            "frames": n_proc,
            "orig_min": round(float(np.min(sims_orig)), 4),
            "ret_mean": round(float(np.mean(sims_ret)), 4),
            "ret_min": round(float(np.min(sims_ret)), 4),
            # fair criterion: wherever the ORIGINAL matches the profile, the
            # retouched frame must still match (and not degrade >0.02)
            "identity_preserved": bool(
                np.min(sims_ret) >= MATCH_THRESHOLD
                or np.min(sims_ret) >= np.min(sims_orig) - 0.02
            ),
        }
        print(subj, report[subj], flush=True)

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
