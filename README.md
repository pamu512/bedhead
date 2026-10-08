# bedhead

**Look presentable on video calls.** Bed Head is a Tier A "good-day" filter: it tracks your face with a 478-point mesh and applies corrective retouch — skin, under-eyes, shine, teeth, hairline — then publishes the result to a virtual camera your meeting app already knows.

> You, but on a good day. Not someone else.

![pipeline](assets/bedhead-architecture.png)

## Status

P0 spike (Python). Tier A retouch only — the generative "good-day re-render" (Tier B) is a later phase; see [`docs/ROADMAP.md`](docs/ROADMAP.md).

- ✅ MediaPipe 478-point face tracking (VIDEO mode, ~5 ms/frame on CPU)
- ✅ Mesh-aware retouch: skin / under-eye / shine / teeth / hairline / soft-light
- ✅ Virtual camera output via OBS (macOS) or native (Windows)
- ✅ Preview window with A/B toggle + keyboard dials
- ✅ Live control panel (`bedhead.panel`) with preset hot-reload
- ⏳ Tier B generative re-render (LivePortrait-class, GPU)
- ⏳ Identity guard (embedding drift cap)
- ⏳ Native macOS app + CMIO camera extension

## Quick start

```bash
# 1) clone and enter
git clone https://github.com/pamu512/bedhead.git
cd bedhead

# 2) install (Python 3.10–3.12, mediapipe constraint)
uv venv && uv pip install -e .
#    (or: python3.12 -m venv .venv && .venv/bin/pip install -e .)

# 1b) macOS screen-recording permission: System Settings → Privacy & Security
#     → Screen Recording → allow Terminal (or your terminal app). Camera permission too.

# 3) run preview only (no OBS needed)
bedhead

# 4) run with virtual camera (macOS: install OBS first — see below)
bedhead --cam

# 5) optional live control panel (separate terminal, sliders apply live)
python -m bedhead.panel
```

First run downloads two MediaPipe task models (~15 MiB total) into `models/` (gitignored).

### macOS virtual camera setup (OBS path)

`pyvirtualcam` on macOS needs OBS Studio with the obs-mac-virtualcam plugin:

1. Install OBS Studio 30+ (brew install --cask obs or obs-28+ with plugin bundled... see below).
2. In OBS: Tools → Start Virtual Camera.
3. bedhead will now find "OBS Virtual Camera". In Zoom/Meet/Teams pick **OBS Virtual Camera** as your camera.

Windows: pyvirtualcam uses the native OBS virtual camera driver (ships with OBS install). No plugin gymnastics.

### Keys (preview window)

| Key | Action |
|-----|--------|
| `q` / `Esc` | quit |
| `space` | toggle original / retouched (A/B) |
| `0`–`9` | set global intensity 0–100% |
| `s` / `e` / `h` / `t` / `l` | skin / under-eye / shine / teeth / soft-light +0.1 (wraps) |

## How it works

```
webcam ──► MediaPipe FaceLandmarker ──► Tier A retoucher ──► virtual camera (OBS)
                (478 pts, ~5 ms)         (skin/eye/shine/      (Zoom/Meet/Teams
                                         teeth/hair/light)      see a webcam)
```

| Stage | Module | What it does | Fail-safe |
|-------|--------|--------------|-----------|
| Track | `bedhead/tracker.py` | 478-point mesh + blendshapes (jawOpen gates teeth) | No face → frame passes through untouched |
| Retouch | `bedhead/retoucher.py` | Feathered, landmark-masked effects; all strengths 0–1, capped | Any effect at 0 → code path skipped |
| Deliver | `bedhead/sinks.py` | Preview window and/or pyvirtualcam sink | vcam open fails → preview-only, never crash |
| Control | `bedhead/cli.py` + `panel.py` | Keyboard dials + Tk sliders; preset JSON hot-reload | — |

Every regional edit is feathered and clamped; nothing drifts: the output is your frame with bounded corrections, never a generated face (that's Tier B, guarded).

## Repo layout

```
bedhead/
├── bedhead/            # Python package
│   ├── cli.py          # entrypoint: bedhead …
│   ├── tracker.py      # MediaPipe face mesh
│   ├── retoucher.py    # Tier A effects
│   ├── sinks.py        # preview + virtual camera
│   ├── panel.py        # Tk live control panel
│   ├── config.py       # Preset dataclass + named presets
│   └── models.py       # one-time model download
├── docs/ROADMAP.md     # phases P0→P3 (from the product brief)
└── assets/             # architecture figure
```

## Configuration

All knobs live in `bedhead/config.py` (`Preset` dataclass). The panel writes `~/.bedhead/preset.json`; the CLI hot-reloads it every second. Named presets: `bedhead --preset subtle` (0.4) or `rescue` (0.8).

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Privacy

Frames never leave the machine. Models download from Google's public MediaPipe store once. No telemetry, no cloud, no analytics. Tier B will process on-device too.

## License

TBD — pick before any public launch. (MIT suggested for the spike; revisit before Series A conversations about the identity-guard IP.)
