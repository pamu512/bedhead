"""Command-line entrypoint.

  bedhead                     preview window + keyboard controls
  bedhead --cam               also send to virtual camera (OBS required on macOS)
  bedhead --no-preview        headless (virtual camera only)
  bedhead --width 1280 --height 720 --fps 30
  bedhead --preset rescue     named preset (subtle|rescue) or JSON file
  bedhead --list-cameras

Keys (preview window):
  q/Esc quit · space toggle original/retouched · 0-9 set intensity
  s skin · e under-eye · h shine · t teeth · l soft-light (each +0.1, wrap)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from . import __version__
from .config import PRESETS, Preset
from .retoucher import apply
from .sinks import PreviewWindow, VirtualCamSink
from .tracker import FaceTracker


def list_cameras() -> None:
    print("Probing cameras 0..5 (something must be readable to confirm)...")
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            found.append((i, w, h))
        cap.release()
    if not found:
        print("No cameras found.")
    for i, w, h in found:
        print(f"  /dev/video{i} (index {i}): {w}x{h}")
    print("Use --camera <index> to pick one.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bedhead", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cam", action="store_true", help="enable virtual camera output")
    ap.add_argument("--no-clothes", action="store_true", help="disable clothes tidy-up (skip segmenter)")
    ap.add_argument("--no-preview", action="store_true", help="disable preview window")
    ap.add_argument("--preset", default=None, help="named preset (subtle|rescue) or JSON path")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--version", action="version", version=f"bedhead {__version__}")
    args = ap.parse_args(argv)

    if args.list_cameras:
        list_cameras()
        return 0

    preset = Preset()
    if args.preset:
        if args.preset in PRESETS:
            for k, v in PRESETS[args.preset].items():
                setattr(preset, k, v)
        else:
            try:
                preset = Preset.load(args.preset)
            except Exception as e:  # noqa: BLE001
                print(f"Could not load preset {args.preset}: {e}", file=sys.stderr)
                return 2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}. Try --list-cameras.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    tracker = FaceTracker()

    segmenter = None
    if not args.no_clothes:
        try:
            from .clothes import ClothesSegmenter

            segmenter = ClothesSegmenter()
            print("[bedhead] clothes tidy-up enabled (segmenter ready)")
        except Exception as e:  # noqa: BLE001
            print(f"[bedhead] clothes tidy-up unavailable ({e.__class__.__name__}); continuing face-only.")
            segmenter = None

    vcam: VirtualCamSink | None = None
    if args.cam:
        try:
            vcam = VirtualCamSink(args.width, args.height, args.fps)
        except Exception as e:  # noqa: BLE001
            print(f"[bedhead] virtual camera unavailable ({e.__class__.__name__}: {e}).")
            print("[bedhead] -> on macOS: install OBS + obs-mac-virtualcam plugin; "
                  "continuing preview-only.")

    preview: PreviewWindow | None = None
    if not args.no_preview:
        preview = PreviewWindow()

    print("[bedhead] running. Keys: q quit · space A/B · 0-9 intensity · "
          "s/e/h/t/l effect dials")

    # FPS stats + panel hot-reload (edits from bedhead.panel land within ~1 s)
    from pathlib import Path as _P
    _preset_path = _P.home() / ".bedhead" / "preset.json"
    _preset_mtime: float = 0.0
    t_last = time.perf_counter()
    t_reload = t_last
    fps_ema = 0.0
    proc_ms_ema = 0.0
    frame_i = 0
    cache: dict = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[bedhead] camera read failed; retrying...")
                time.sleep(0.05)
                continue
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            t0 = time.perf_counter()
            face = tracker.detect(frame, frame_i * 1000 // max(args.fps, 1))
            out = apply(frame, face, preset, cache)
            if segmenter is not None:
                from .clothes import tidy

                ts_ms = frame_i * 1000 // max(args.fps, 1)
                cmask = segmenter.clothes_mask_cached(frame, ts_ms)
                out = tidy(out, cmask, preset)
            proc_ms = (time.perf_counter() - t0) * 1000
            proc_ms_ema = proc_ms if frame_i == 0 else proc_ms_ema * 0.9 + proc_ms * 0.1

            if vcam is not None:
                vcam.send(out)

            if preview is not None:
                hud = (
                    f"bedhead {__version__} | {fps_ema:5.1f} fps | track+retouch {proc_ms_ema:4.1f} ms"
                    f" | face {'LOST (passthrough)' if face is None else 'ok'}"
                    f" | {preset.describe()}"
                )
                key = preview.show(out, hud=hud)
                if key in ("q", "\x1b", "Q"):
                    break
                if key == " ":
                    preset.show_original = not preset.show_original
                elif key and key.isdigit():
                    preset.intensity = int(key) / 9.0
                elif key in ("s", "S"):
                    preset.skin = (preset.skin + 0.1) % 1.1
                elif key in ("e", "E"):
                    preset.under_eye = (preset.under_eye + 0.1) % 1.1
                elif key in ("h", "H"):
                    preset.shine = (preset.shine + 0.1) % 1.1
                elif key in ("t", "T"):
                    preset.teeth = (preset.teeth + 0.1) % 1.1
                elif key in ("l", "L"):
                    preset.soft_light = (preset.soft_light + 0.1) % 1.1

            # housekeeping
            now = time.perf_counter()
            if now - t_reload > 1.0 and _preset_path.exists():
                t_reload = now
                mtime = _preset_path.stat().st_mtime
                if mtime != _preset_mtime:
                    _preset_mtime = mtime
                    try:
                        preset = Preset.load(str(_preset_path))
                        print(f"[bedhead] preset reloaded: {preset.describe()}")
                    except Exception:  # noqa: BLE001
                        pass
            inst = 1.0 / max(now - t_last, 1e-6)
            fps_ema = inst if frame_i == 0 else fps_ema * 0.9 + inst * 0.1
            t_last = now
            frame_i += 1
            if frame_i % 300 == 0:
                print(f"[bedhead] {fps_ema:5.1f} fps | {proc_ms_ema:4.1f} ms | "
                      f"frames {frame_i} | vcam {vcam.frames_sent if vcam else '-'}")
    except KeyboardInterrupt:
        print("\n[bedhead] interrupted")
    finally:
        cap.release()
        tracker.close()
        if segmenter is not None:
            segmenter.close()
        if vcam is not None:
            vcam.close()
        if preview is not None:
            preview.close()
        print(f"[bedhead] bye. stats: {fps_ema:.1f} fps avg, {proc_ms_ema:.1f} ms avg pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
