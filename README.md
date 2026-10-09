# bedhead

**Look presentable on video calls.** Bed Head is a Tier A "good-day" filter: it tracks your face with a 478-point mesh and applies corrective retouch — skin, under-eyes, shine, teeth, hairline — then publishes the result to a virtual camera your meeting app already knows.

> You, but on a good day. Not someone else.

![pipeline](assets/bedhead-architecture.png)

## Status

P0.5 (Python). Tier A retouch rebuilt on a research-grade quality engine, plus the parity features (background modes, person relight, eye light); the generative "good-day re-render" (Tier B) is a later phase; see [`docs/ROADMAP.md`](docs/ROADMAP.md).

- ✅ MediaPipe 478-point face tracking (VIDEO mode, ~5 ms/frame on CPU), one-euro smoothed
- ✅ Quality-engine retouch: guided-filter frequency separation, LAB pipeline, local-percentile shine, hysteresis teeth gate, selective sharpening (~9 ms/frame for all face effects at 720p on Apple Silicon)
- ✅ Person segmentation (selfie-multiclass): true person masks, EMA-smoothed, every-3rd-frame
- ✅ Background blur / darken (Zoom Portrait / NVIDIA-class, feathered composite)
- ✅ Studio Light: relight the person only, background untouched (Apple Continuity-class)
- ✅ Eye light: landmark-gated brightness for an awake look
- ✅ Identity preservation verified: 43/43 VidTIMIT subjects, 3,512 frames, retouched faces still match their profile picture (mean SFace cosine 0.83 vs 0.363 threshold; no frame degrades below its original)
- ✅ Virtual camera output via OBS (macOS) or native (Windows)
- ✅ Preview window with A/B toggle + keyboard dials (incl. k/i/b/n for the new effects)
- ✅ Live control panel (`bedhead.panel`) with preset hot-reload
- ✅ Headless test suite + CI (ruff + pytest, 3.10–3.13)
- ⏳ Tier B generative re-render (LivePortrait-class; research says Core ML/ANE only, not CPU)
- ⏳ Identity guard runtime drift cap (admission check already shipped, see below)
- ⏳ Native macOS app + CMIO camera extension

## Quick start

```bash
# 1) clone and enter
git clone https://github.com/pamu512/bedhead.git
cd bedhead

# 2) install (Python 3.10–3.13)
uv venv && uv pip install -e .
#    (or: python3.12 -m venv .venv && .venv/bin/pip install -e .)

# 1b) macOS screen-recording permission: System Settings → Privacy & Security
#     → Screen Recording → allow Terminal (or your terminal app). Camera permission too.

# 3) run preview only (no OBS needed)
bedhead

# 3b) face-only, skip the clothes segmenter
bedhead --no-clothes

# 4) run with virtual camera (macOS: install OBS first — see below)
bedhead --cam

# 5) optional live control panel (separate terminal, sliders apply live)
python -m bedhead.panel
```

First run downloads two MediaPipe models (~20 MiB total: face landmarker 3.7 MiB + selfie segmenter 16.4 MiB) into your user cache directory (`~/Library/Caches/bedhead` on macOS, `~/.cache/bedhead` on Linux), each verified against a pinned sha256. The segmenter is only fetched when a segmentation feature (background/studio light) is first used.

### macOS virtual camera setup (OBS path)

`pyvirtualcam` on macOS needs OBS Studio with the obs-mac-virtualcam plugin:

1. Install OBS Studio 30+ (brew install --cask obs or obs-28+ with plugin bundled... see below).
2. In OBS: Tools → Start Virtual Camera.
3. bedhead will now find "OBS Virtual Camera". In Zoom/Meet/Teams pick **OBS Virtual Camera** as your camera.

Windows: pyvirtualcam uses the native OBS virtual camera driver (ships with OBS install). No plugin gymnastics.

### Keys (preview window)

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `space` | toggle original / retouched in the **preview only** (the virtual camera keeps receiving the retouched frame) |
| `0`–`9` | set global intensity (0 to 100% in 1/9 steps) |
| `s` / `e` / `h` / `t` / `l` | skin / under-eye / shine / teeth / soft-light +0.1 (wraps) |
| `k` / `i` | studio light / eye light +0.1 (wraps) |
| `b` / `n` | background strength +0.2 (wraps) / cycle background mode off→blur→dark |

Keyboard edits are saved to `~/.bedhead/preset.json` and the Tk panel picks them up (and vice versa).

### Named presets

| Preset | What it does |
| --- | --- |
| `subtle` | light retouch (intensity 0.4) |
| `rescue` | full retouch (intensity 0.8) |
| `studio` | retouch + studio light + eye light + blurred background |
| `focus` | light retouch + darkened background (all attention on you) |

### Reference photo (identity-guarded)

Tier B features take a reference photo, and it can come from your gallery:

```
bedhead --reference ~/Pictures/good-day.jpg
```

The reference is only used after it passes an on-device admission check: bedhead samples ~15 live frames, embeds the face on camera and the face in the photo (ArcFace), and requires cosine similarity >= 0.40 (calibrated: same-person photos score 0.73+, different people score below 0.1). A photo of someone else is rejected and the run continues Tier-A-only. Requires the `guard` extra: `pip install 'bedhead[guard]'`. Once Tier B lands, its output will additionally be drift-capped against the admitted reference (fail-safe to Tier A).

With `--auto-match`, the admitted reference also tunes the Tier A effects: bedhead measures the exposure/warmth/sharpness gap between the live feed and the reference photo and derives `soft_light` / `studio_light` / `under_eye` / `skin` strengths that move your live look toward the photo's look (classical effects only, nothing generative):

```
bedhead --reference ~/Pictures/good-day.jpg --auto-match
```

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

## Quality engine (v2)

The retouch core was rebuilt around research-grounded techniques (see `bedhead/quality.py`):

| Technique | What it does | Source |
| --- | --- | --- |
| Guided-filter frequency separation | smooths only mid-frequency blemishes on the L channel; pores and edges survive (no plastic look) | He et al., ECCV 2010 |
| One-Euro landmark filtering | kills landmark jitter (the #1 "cheap filter" tell) with near-zero lag on motion | Casiez & Roussel, CHI 2012 |
| LAB color pipeline | all corrections on luminance; chroma untouched (except deliberate teeth desat) | standard pro-retouch practice |
| Local-percentile shine | specular highlights found vs the LOCAL smooth base, works across skin tones | intrinsic-decomposition practice |
| Hysteresis gates | jaw-open Schmitt trigger stops the teeth effect flickering at half-open | classical |
| Selective unsharp | eyes/lips/brows sharpened via the high-frequency residual, skin left smooth | pro workflow |

Optional segmentation effects (`bedhead.segmenter`, downloads its model on first use): background blur/darken, Studio Light (person-only relight), eye light, and true-skin masking that refines the landmark oval (handles hair strands, glasses).

## Identity preservation benchmark

`scripts/identity_check.py` verifies the retouch never changes who you are. On the Kaggle [VidTIMIT](https://www.kaggle.com/datasets/crazyt/vidtimit-audiovideo-dataset) dataset (43 subjects; first frame per subject acts as the profile picture, all other clips are retouched at "rescue" strength):

- mean SFace cosine similarity retouched-vs-profile: **0.83** (match threshold 0.363)
- every one of the 3,512 processed frames scores at least as well as its unretouched original (max degradation 0.03, within noise)

Run it yourself:

```bash
kaggle datasets download crazyt/vidtimit-audiovideo-dataset -p vidtimit --unzip
python scripts/identity_check.py --frames-root vidtimit --preset rescue
```

## Repo layout

```
bedhead/
├── bedhead/            # Python package
│   ├── cli.py          # entrypoint: bedhead …
│   ├── tracker.py      # MediaPipe face mesh (one-euro smoothed)
│   ├── retoucher.py    # Tier A effects (LAB + guided-filter engine)
│   ├── quality.py      # one-euro filter, hysteresis, frequency separation
│   ├── lighting.py     # studio light / eye light / background modes
│   ├── segmenter.py    # person/skin/hair segmentation (amortized)
│   ├── guard.py        # ArcFace identity guard (reference admission)
│   ├── sinks.py        # preview + virtual camera
│   ├── panel.py        # Tk live control panel
│   ├── config.py       # Preset dataclass + named presets
│   └── models.py       # one-time model download (sha256-pinned)
├── scripts/identity_check.py  # identity-preservation benchmark (VidTIMIT)
├── tests/              # headless test suite (+ live_camera marker)
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

MIT (see [LICENSE](LICENSE)). The identity-guard IP planned for Tier B will be reviewed separately before any Series A conversations.
