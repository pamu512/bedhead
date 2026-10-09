"""Command-line entrypoint.

  bedhead                     preview window + keyboard controls
  bedhead --cam               also send to virtual camera (OBS required on macOS)
  bedhead --no-preview        headless (virtual camera only)
  bedhead --width 1280 --height 720 --fps 30
  bedhead --preset rescue     named preset (subtle|rescue) or JSON file
  bedhead --list-cameras

Keys (preview window):
  q/Esc quit · space toggle original/retouched (preview only; the virtual
  camera always receives the retouched frame) · 0-9 set intensity
  s skin · e under-eye · h shine · t teeth · l soft-light (each +0.1, wrap)

Keyboard edits are persisted to ~/.bedhead/preset.json so the Tk panel and
the CLI share one source of truth.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .autotune import LookTracker
from .config import BACKGROUND_MODES, PRESETS, Preset
from .guard import LIVE_SAMPLES, IdentityGuard
from .retoucher import apply
from .segmenter import Segmenter
from .sinks import PreviewWindow, VirtualCamSink
from .tracker import FACE_OVAL, FaceTracker

PRESET_PATH = Path.home() / ".bedhead" / "preset.json"
MAX_CONSECUTIVE_READ_FAILURES = 100  # ~5 s of retries before giving up


def _camera_names() -> dict[int, str]:
    """Best-effort human-readable camera names (macOS: AVFoundation via pyobjc)."""
    if sys.platform == "darwin":
        try:
            import AVFoundation

            devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
                AVFoundation.AVMediaTypeVideo
            )
            return {i: str(d.localizedName()) for i, d in enumerate(devices)}
        except (ImportError, AttributeError, RuntimeError):
            # pyobjc optional; probing fallback used below
            pass
    return {}


def list_cameras() -> None:
    print("Probing cameras 0..5 (something must be readable to confirm)...")
    names = _camera_names()
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
        label = names.get(i, f"camera {i}")
        print(f"  [{i}] {label}: {w}x{h}")
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
    ap.add_argument("--preset", default=None, help="named preset (subtle|rescue|studio|focus) or JSON path")
    ap.add_argument("--reference", default=None, metavar="PHOTO",
                    help="reference photo for identity-guarded features (gallery upload OK; "
                         "must match the face on camera to be used)")
    ap.add_argument("--auto-match", action="store_true",
                    help="with --reference: derive effect strengths from the reference look "
                         "(exposure/warmth/sharpness match) and apply them")
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
    look_tracker: LookTracker | None = None  # set when --auto-match is admitted
    reference_img: np.ndarray | None = None  # the admitted reference photo
    cm_cache: dict = {}  # reference-stats cache for color_match (per run)

    # --- reference photo admission (gallery upload allowed, identity-gated) ---
    if args.reference:
        photo = cv2.imread(args.reference)
        if photo is None:
            print(f"[bedhead] could not read reference photo {args.reference}", file=sys.stderr)
            return 2
        try:
            guard = IdentityGuard()
        except RuntimeError as e:
            print(f"[bedhead] {e}", file=sys.stderr)
            return 2
        ref_emb = guard.embed(photo)
        if ref_emb is None:
            print("[bedhead] no face found in the reference photo; "
                  "pick a clear, front-facing shot.", file=sys.stderr)
            return 2
        if guard.last_pose and abs(guard.last_pose[1]) > 35:
            print(f"[bedhead] note: the reference face is turned "
                  f"({guard.last_pose[1]:.0f} deg yaw); heavily angled photos "
                  "often fail the match check. A front-facing photo of the same "
                  "person matches much more reliably.", file=sys.stderr)
        print(f"[bedhead] sampling {LIVE_SAMPLES} live frames to check the reference "
              f"matches the face on camera ...")
        live_embs: list[np.ndarray] = []
        live_frames: list[np.ndarray] = []
        attempts = 0
        while len(live_embs) < LIVE_SAMPLES and attempts < LIVE_SAMPLES * 6:
            attempts += 1
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if frame.shape[:2] != (args.height, args.width):
                frame = cv2.resize(frame, (args.width, args.height))
            e = guard.embed(frame)
            if e is not None:
                live_embs.append(e)
                live_frames.append(frame)
        result = guard.admit(ref_emb, live_embs)
        if not result.admitted:
            print(f"[bedhead] reference rejected: {result.reason}", file=sys.stderr)
            print("[bedhead] continuing WITHOUT the reference (Tier A only).", file=sys.stderr)
        else:
            print(f"[bedhead] reference admitted: {result.reason}")
            reference_img = photo
            if args.auto_match:
                # reference-guided autotune: one-shot suggestions at startup,
                # then continuous ambient adaptation while the call runs
                from dataclasses import asdict

                from .autotune import LookTracker, apply_autotune, autotune

                live_sample = live_frames[-1] if live_frames else None
                if live_sample is not None:
                    at = autotune(live_sample, photo)
                    for note in at.notes:
                        print(f"[bedhead] auto-match: {note}")
                    merged = apply_autotune(asdict(preset), at)
                    for k, v in merged.items():
                        setattr(preset, k, v)
                    # color match rides along with auto-match at a fixed
                    # moderate strength (classical Reinhard transfer)
                    preset.color_match = max(preset.color_match, 0.8)
                    print(f"[bedhead] auto-match applied (color_match "
                          f"{preset.color_match:.2f}): {preset.describe()}")
                    look_tracker = LookTracker(photo)
                    look_tracker.prime(live_sample)
                    print("[bedhead] auto-match: continuous mode on "
                          "(ambient adaptation every ~2 s)")
                    try:
                        preset.save(str(PRESET_PATH))
                    except OSError as e:
                        print(f"[bedhead] could not save preset: {e}")

    segmenter: Segmenter | None = None

    def _seg_wanted(p: Preset) -> bool:
        return (p.background_strength > 0 and p.background_mode != "off") or p.studio_light > 0

    def _get_segmenter() -> Segmenter | None:
        """Lazily construct the segmenter (downloads model on first use)."""
        nonlocal segmenter
        if not _seg_wanted(preset):
            return None
        if segmenter is None:
            try:
                segmenter = Segmenter()
                print("[bedhead] segmentation active (person masks for background/studio light)")
            except Exception as e:  # noqa: BLE001 - degrade to no segmentation
                print(f"[bedhead] segmentation unavailable ({e.__class__.__name__}: {e}); "
                      "background/studio-light disabled this run.")
                return None
        return segmenter

    clothes_seg = None

    def _get_clothes():
        """Lazily construct the clothes segmenter (tidy-up + logo blur)."""
        nonlocal clothes_seg
        if args.no_clothes:
            return None
        if clothes_seg is None:
            try:
                from .clothes import ClothesSegmenter

                clothes_seg = ClothesSegmenter()
                print("[bedhead] clothes tidy-up enabled (segmenter ready)")
            except Exception as e:  # noqa: BLE001
                print(f"[bedhead] clothes tidy-up unavailable ({e.__class__.__name__}); "
                      "continuing face-only.")
                return None
        return clothes_seg

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

    if vcam is None and preview is None:
        print("[bedhead] no output sink available (virtual camera failed, preview disabled); "
              "nothing to do.", file=sys.stderr)
        cap.release()
        tracker.close()
        return 1

    print("[bedhead] running. Keys: q/Esc quit · space A/B · 0-9 intensity · "
          "s/e/h/t/l dials · m color-match · k studio · i eye-light · b bg strength · n bg mode")

    # FPS stats + panel hot-reload (edits from bedhead.panel land within ~1 s)
    _preset_mtime: float = 0.0
    if PRESET_PATH.exists():
        _preset_mtime = PRESET_PATH.stat().st_mtime
    t_last = time.perf_counter()
    t_reload = t_last
    fps_ema = 0.0
    proc_ms_ema = 0.0
    frame_i = 0
    read_failures = 0
    key: str | None = None

    def persist() -> None:
        """Save keyboard edits so the panel sees them; bump mtime to skip self-reload."""
        nonlocal _preset_mtime
        try:
            preset.save(str(PRESET_PATH))
            _preset_mtime = PRESET_PATH.stat().st_mtime
        except OSError as e:
            print(f"[bedhead] could not save preset: {e}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    print(f"[bedhead] camera {args.camera} produced {read_failures} consecutive "
                          "read failures; exiting.", file=sys.stderr)
                    return 1
                time.sleep(0.05)
                continue
            read_failures = 0
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            t0 = time.perf_counter()
            face = tracker.detect(frame, frame_i * 1000 // max(args.fps, 1))
            seg = _get_segmenter()
            if seg is not None:
                oval = (face.landmarks[list(FACE_OVAL), :2].astype(int)
                        if face is not None else None)
                seg.tick(frame, frame_i * 1000 // max(args.fps, 1), face_oval_pts=oval)
            # A/B is preview-only: the virtual camera always gets the retouched
            # frame, so toggling it mid-call can never leak the unretouched feed.
            effect_preset = preset if not preset.show_original else replace(
                preset, show_original=False
            )
            out = apply(frame, face, effect_preset, segmenter=seg,
                        reference_bgr=reference_img, _cm_cache=cm_cache)
            # clothes tidy-up + logo blur run on the retouched frame
            cseg = _get_clothes()
            if cseg is not None and (preset.clothes > 0 or preset.stain > 0 or preset.logo_blur > 0):
                from .clothes import tidy

                ts_ms = frame_i * 1000 // max(args.fps, 1)
                cmask = cseg.clothes_mask_cached(frame, ts_ms)
                out = tidy(out, cmask, preset)
            proc_ms = (time.perf_counter() - t0) * 1000
            proc_ms_ema = proc_ms if frame_i == 0 else proc_ms_ema * 0.9 + proc_ms * 0.1

            if vcam is not None:
                vcam.send(out)

            if preview is not None:
                shown = frame if preset.show_original else out
                hud = (
                    f"bedhead {__version__} | {fps_ema:5.1f} fps | track+retouch {proc_ms_ema:4.1f} ms"
                    f" | face {'LOST (passthrough)' if face is None else 'ok'}"
                    f" | {'A/B: ORIGINAL (preview only)' if preset.show_original else ''}"
                    f" | {preset.describe()}"
                )
                key = preview.show(shown, hud=hud)
                if key in ("q", "Q", "\x1b"):
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
                elif key in ("m", "M"):
                    preset.color_match = (preset.color_match + 0.2) % 1.2
                elif key in ("b", "B"):
                    preset.background_strength = (preset.background_strength + 0.2) % 1.2
                elif key in ("k", "K"):
                    preset.studio_light = (preset.studio_light + 0.1) % 1.1
                elif key in ("i", "I"):
                    preset.eye_light = (preset.eye_light + 0.1) % 1.1
                elif key in ("n", "N"):
                    modes = BACKGROUND_MODES
                    preset.background_mode = modes[(modes.index(preset.background_mode) + 1) % len(modes)]
                else:
                    key = None
                if key is not None:
                    persist()

            # housekeeping
            now = time.perf_counter()
            if look_tracker is not None and look_tracker.tick(frame):
                for k, v in look_tracker.current.items():
                    setattr(preset, k, v)
                print("[bedhead] auto-match adapted: "
                      + " ".join(f"{k}={v:.2f}" for k, v in look_tracker.current.items()))
            if now - t_reload > 1.0 and PRESET_PATH.exists():
                t_reload = now
                mtime = PRESET_PATH.stat().st_mtime
                if mtime != _preset_mtime:
                    _preset_mtime = mtime
                    try:
                        preset = Preset.load(str(PRESET_PATH))
                        print(f"[bedhead] preset reloaded: {preset.describe()}")
                    except Exception as e:  # noqa: BLE001
                        print(f"[bedhead] preset reload failed ({e}); keeping current preset")
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
