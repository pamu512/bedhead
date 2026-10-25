# Measured results (provenance + numbers)

Every number the README/ROADMAP quotes, where it came from, and the exact
command to reproduce it. Nothing here was produced by running competitor
software; benchmark targets are engineering targets authored from public
claims (see `bedhead/benchmark.py` docstring).

## Latency profile (all effects on, 720p)

scripts/latency_profile.py, M-series CPU, 30 frames warm: track 5.4 ms,
segmentation 6.6 mean (19.4 p95 on refresh frames), retouch 20.2, clothes
15.0 amortized -> end-to-end 47.1 ms mean (21 fps ceiling with EVERY effect
on incl. blur+studio+vibrance+clothes). Face-stack-only configs run 6-9 ms
(110+ fps). 2026-10-25 perf round: blendLinear composites + LUT warmth
(lighting) and crease-delta caching (clothes) took the all-on profile from
63.2 -> 47.1 ms.

## Live quality scoreboard (8-metric benchmark)

2026-10-25: under-eye v4 (deficit cap 85, k to 1.05, mid-feather full alpha,
cheek ceiling) + shine/under-eye ordering fix (shine ran AFTER the dodge and
re-compressed the lifted trough). Pinned headless in tests/test_beyond_parity.py;
scripts/beyond_parity_check.py reproduces the numbers below.

Best clean seated session, 2026-10-09 (macOS, MacBook Pro Camera, 720p,
reference photo admitted, LookTracker continuous adaptation):

| metric | value | target | verdict |
| --- | --- | --- | --- |
| subject_pop | +63.6 (median bg 204 -> 44, face untouched 108/108) | >= +15 | PASS |
| skin_texture_hf_pct | 96-125% across sessions | >= 90% | PASS |
| color_fidelity_dE | 0.95-1.5 | < 8 | PASS |
| exposure_gap | 0.85-7.0 | < 8 | PASS |
| saturation_delta | 1.5-4.7 | < 10 | PASS |
| temporal_pump | 0.02-0.03 | < 2.0 | PASS |
| latency_ms | 31-45 (steady-state; first-run warm-up excluded) | < 45 | PASS (borderline) |
| under_eye_gap | v3: 19-69 depending on room lighting (raw deficit 63-82 L*) | < 10 | v4: CLOSED in code — synthetic bench 46/64/82-level deficits -> 0/0/0 gap; real-photo check: raw 85 -> 14 at default-ish dials, -5..1 at full dial (scripts/beyond_parity_check.py). Live-session scoreboard update pending. |

Reproduce: capture raw+processed frames against your reference and score
them with `bedhead.benchmark.benchmark()` (methodology: median of the last
5 settled samples, face-oval masks, reference face detected + tightened).
Sessions vary with lighting; the range column is honest across runs.

## Identity preservation (VidTIMIT)

43/43 subjects, 3,512 frames (stride 10), mean SFace cosine 0.83 (threshold
0.363). Per-frame worst drop vs original: <= 0.03 (the script tolerates up
to 0.02 drift per subject's worst frame by design).

Reproduce:
```
python scripts/identity_check.py --frames-root <vidtimit-extracted> --stride 10
```
VidTIMIT frames are not redistributed; download requires an academic
request. The script writes per-subject JSON to --out.

## Fairness (guard + tracker across skin tones)

99 photos / 15 people (east/south Asian, African descent), video-call
framing normalization. Guard admission: passes all groups (worst
same-person 0.594 vs 0.40 gate; best cross-person 0.110). Tracker lock:
62% on the african cohort vs 87-89% others; misses correlate with extreme
yaw (51-59 deg) or dark captures (L* 69-76). CLAHE retry did not recover
them. Mitigation shipped: tracker gap-bridging (velocity-extrapolated hold
with a 10-frame confidence ramp, effects fade via the ramp; re-detection
wins instantly; tests/test_beyond_parity.py). Live cohort re-measure pending.

Reproduce:
```
python scripts/fetch_fairness_fixtures.py   # populates /tmp/bh_fairness (Commons, research UA)
pytest tests/test_fairness.py               # skips without the fixtures
```

## Tier B spike (IN Swapper 128)

- CoreML-assisted (CPU+ANE partitioning): ~64-76 ms/frame warm
- Pure CPU: 205 ms; fp16 conversion: measured regression (not used)
- Own-face drift similarity 0.947 (cap 0.35); foreign reference rejected
  at admission with similarity 0.103
- Frequency composite: cheek HF equal to the aligned 128px crop's
  (native pore-level detail beyond 128px is not invented)
- 256px hyperswap: closed, see TIERB-256-EVAL.md

Reproduce: `bedhead --reference you.jpg --tier-b` (guard extra + local
`inswapper_128.onnx` in the model cache).

## Color match

Reinhard LAB transfer, face oval, chroma-clamped (MAX_CHROMA_SHIFT 30).
Look-gap closure on the 0.55x-exposure test frame: 45% first measurement,
42% on remeasure (honest range: ~42-45%). Identity stays ~0.90.
