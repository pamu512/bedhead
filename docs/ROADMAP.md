# Bed Head — Build Roadmap

Full background: see the product & engineering brief (repo `assets/` + local `bedhead-blueprint.html`).

## P0 — Spike (now)

**Goal:** Tier A retouch, real camera in → virtual camera out, 30 fps @720p.

- [x] Repo + CI-less spike structure
- [x] MediaPipe tracking (VIDEO mode, blendshapes for jawOpen/blink)
- [x] Tier A effects: skin, under-eye, shine, teeth, hairline, soft-light
- [x] Preview + keyboard A/B + dials
- [x] Virtual camera sink (OBS path on macOS)
- [x] Tk control panel with hot-reload
- [ ] Field-test on real calls (Zoom/Meet/Teams)
- [ ] Benchmark: sustained fps on 3-year-old laptop, effect strengths at defaults

**Exit criteria:** side-by-side A/B demo makes people say "left one, please."

## P1 — Tier B demo (4–8 weeks)

**Goal:** "good-day me" — generative re-render from one reference photo, identity-guarded.

- [ ] Onboarding: guided selfie capture (liveness via camera, no gallery)
- [ ] LivePortrait (or PersonaLive-class) inference on GPU, ~12–30 ms/frame
- [ ] Identity guard: InsightFace embeddings, drift cap, temporal smoothing, fail-safe to Tier A
- [ ] Blend dial: Tier A ↔ Tier B continuous mix

**Exit criteria:** 60-second continuous demo, expressions live, no waxiness.

## P2 — Alpha native app (2–3 months)

- [ ] Swift app, Core ML/MPS inference
- [ ] CMIO camera extension (App Store-eligible virtual camera)
- [ ] Encrypted on-device preset store; uninstall = gone
- [ ] Optional status indicator + one-click kill switch
- [ ] Notarized Developer ID build

**Exit criteria:** clean-machine install < 5 min, week of daily-driver dogfood, ~0 crashes.

## P3 — Windows + closed beta (parallel)

- [ ] ONNX Runtime + DirectShow/MediaFoundation virtual camera driver
- [ ] EV-signed installer
- [ ] Fallback matrix (GPU→CPU, Tier B→A) verified on Intel iGPU
- [ ] 50-user remote-worker beta

## Non-goals (standing)

- No cloud processing. Ever.
- No face other than the user's (liveness capture only, no gallery uploads).
- No auto-apply at OS level — user taps in per call.
