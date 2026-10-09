# Bed Head: Build Roadmap

Full background: see the product & engineering brief (`docs/PRODUCT-BRIEF.md.html`).

## P0 / P0.5: Spike + quality engine (SHIPPED)

**Goal:** Tier A retouch, real camera in → virtual camera out, 30 fps @720p.

Shipped and measured:

- [x] MediaPipe 478-pt tracking (VIDEO mode, one-euro smoothed, ~5 ms/frame)
- [x] Quality-engine retouch: guided-filter frequency separation, LAB
      pipeline, guided-filter-base shine compression, hysteresis teeth gate (~9 ms/frame)
- [x] Reference-guided: identity-gated gallery reference (ArcFace cosine
      >= 0.40 admission), autotune, continuous ambient adaptation
      (LookTracker), Reinhard color match, face_lift exposure closer
- [x] Person segmentation: background blur/darken, studio light, subject
      pop with closed-loop background control (camera AE coupling measured
      and documented; built-in Mac cameras expose NO AE control via
      AVFoundation; see `bedhead/aelock.py`)
- [x] Under-eye correction: convex-hull tear-trough band, darkness-
      proportional dodge, measured (was a 28-px no-op mask)
- [x] Detail boost + vibrance (face-oval ROI): ~114-125% HF retention
- [x] Clothes tidy-up, preview A/B + dials + reference PiP, Tk panel
- [x] Virtual camera sink (macOS: obs-mac-virtualcam; Windows: pyvirtualcam on the OBS driver shipped with OBS Studio)
- [x] Headless suite + CI (ruff + pytest, 3.10-3.13) + binary
      build with clean-machine smoke ([docs/PACKAGING.md](PACKAGING.md))
- [x] 8-metric quality benchmark (`bedhead/benchmark.py`) vs
      competitor-derived targets; best clean live session 6/8 PASS with
      color dE 1.3, saturation 2.2, exposure 7.0, texture 96-125%,
      temporal pump 0.02-0.03, latency 31-45 ms
- [x] Fairness checks for east/south Asian and African-descent groups
      when local fixtures are present (tests/test_fairness.py). Extreme
      yaw is still open (see P1).
- [x] MIT license

## P1: Tier B demo (in progress)

**Goal:** "good-day me": generative re-render from one reference photo, identity-guarded.

- [x] IN Swapper 128 spike under the full guard contract (admitted
      reference only, drift cap 0.35 every 10 frames, progressive
      fail-safe: blend halving before disable)
- [x] Frequency-separable composite: low-freq identity from the swap,
      native-res high-freq texture from the camera (HF = raw exactly)
- [x] CoreML-assisted inference profiling (~64-76 ms/frame CPU+ANE
      partitioning; pure CPU 205 ms; fp16 a measured regression)
- [ ] Real-time Tier B (< 45 ms/frame): needs a full ANE/GPU engine port
- [ ] 256px-class model: hyperswap_1a_256 evaluated and CLOSED (46x too
      slow on CPU + embedding-space mismatch; see
      [TIERB-256-EVAL.md](TIERB-256-EVAL.md) for reopen conditions)
- [ ] Fairness: tracker-lock recovery at extreme yaw (one-euro bridging /
      temporal retry)

**Exit criteria:** 60-second continuous demo, expressions live, no waxiness, real-time.

## P2: Alpha native app (2–3 months)

- [ ] Swift app, Core ML/MPS inference
- [ ] CMIO camera extension (App Store-eligible virtual camera)
- [ ] Encrypted on-device preset store; uninstall = gone
- [ ] Optional status indicator + one-click kill switch
- [ ] Notarized Developer ID build (script exists: `scripts/notarize.sh`)

**Exit criteria:** clean-machine install < 5 min, week of daily-driver dogfood, ~0 crashes.

## P3: Windows + closed beta (parallel)

- [ ] ONNX Runtime + DirectShow/MediaFoundation virtual camera driver
- [ ] EV-signed installer
- [ ] Fallback matrix (GPU→CPU, Tier B→A) verified on Intel iGPU
- [ ] 50-user remote-worker beta

## Non-goals (standing)

- No cloud processing. Ever.
- No rendering someone else's face: gallery uploads are allowed as reference
  photos, but only if they pass the ArcFace admission check against the live
  face on camera (similarity gate; a photo of someone else is rejected).
- No auto-apply at OS level: the user taps in per call.
