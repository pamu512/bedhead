"""Live 8-metric benchmark: capture raw + full-pipeline output, score vs reference."""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/Users/pamu/.hermes/profiles/cursor-grok-triage/cache/scratch/bedhead")

from bedhead.autotune import LookTracker, autotune
from bedhead.benchmark import BenchReport, benchmark
from bedhead.cli import _face_bbox_mask
from bedhead.config import Preset
from bedhead.guard import IdentityGuard
from bedhead.retoucher import apply as retouch
from bedhead.segmenter import Segmenter
from bedhead.tracker import FACE_OVAL, FaceTracker

REF = cv2.imread("/tmp/bedhead_reference.jpg")
assert REF is not None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

tr = FaceTracker()
seg = Segmenter()
guard = IdentityGuard()
last_autotune = 0.0
# production path: LookTracker adapts soft_light/studio_light/background_darken
look: LookTracker | None = None
cur_face = None

samples = []
latencies = []
temporal = []
prev_warmth = None
t_end = time.time() + 45
p = Preset()
p.intensity = 0.8

warm = 0
while warm < 8 and time.time() < t_end:
    ok, frame = cap.read()
    if not ok:
        continue
    face = tr.detect(frame, timestamp_ms=int(time.monotonic() * 1000))
    if face is None:
        continue
    now = time.monotonic()
    if now - last_autotune > 1.0:
        fb = _face_bbox_mask(guard, frame)
        rb = _face_bbox_mask(guard, REF)
        sug = autotune(frame, REF, live_face_mask=fb, reference_face_mask=rb, face=face)
        for k, v in sug.preset_delta.items():
            if hasattr(p, k):
                setattr(p, k, v)
        last_autotune = now
    warm += 1

while time.time() < t_end and len(samples) < 10:
    ok, frame = cap.read()
    if not ok:
        continue
    t0 = time.perf_counter()
    face = tr.detect(frame, timestamp_ms=int(time.monotonic() * 1000))
    if face is None:
        continue
    cur_face = face
    now = time.monotonic()
    if now - last_autotune > 2.0:
        fb = _face_bbox_mask(guard, frame)
        rb = _face_bbox_mask(guard, REF)
        sug = autotune(frame, REF, live_face_mask=fb, reference_face_mask=rb, face=face)
        for k, v in sug.preset_delta.items():
            if hasattr(p, k):
                setattr(p, k, v)
        last_autotune = now
        if look is None:
            look = LookTracker(REF, face_mask_fn=lambda: fb,
                               face_fn=lambda: cur_face)
            look.prime(frame)
        else:
            look.tick(frame, now=now)
    if look is not None:
        for k, v in look.current.items():
            setattr(p, k, v)
    seg.tick(frame, int(time.monotonic() * 1000), face_oval_pts=None)
    out = retouch(frame, face, p, segmenter=seg)
    dt = (time.perf_counter() - t0) * 1000
    latencies.append(dt)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    pts = face.landmarks[list(FACE_OVAL), :2]
    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
    warmth = float(lab[y0:y1, x0:x1, 2].mean())
    if prev_warmth is not None:
        temporal.append(abs(warmth - prev_warmth))
    prev_warmth = warmth

    samples.append((frame.copy(), out.copy(), face))

cap.release()

if not samples:
    print("NO SAMPLES: no face detected during window")
    sys.exit(1)

steady_samples = samples[-5:]
rows = []
for raw, out, face in steady_samples:
    rep = benchmark(raw, out, REF, face,
                    latency_ms=float(np.median(latencies[3:] if len(latencies) > 4 else latencies)),
                    temporal_deltas=temporal)
    rows.append(rep)
rep = BenchReport()
for i, r0 in enumerate(rows[0].results):
    vals = [r.results[i].value for r in rows]
    med = float(np.median(vals))
    target = r0.target
    passed = med < target if r0.name != "subject_pop" else med >= target
    if r0.name == "skin_texture_hf_pct":
        passed = med >= target
    rep.add(r0.name, med, target, passed,
            detail="median of " + str(len(rows)) + " settled samples")
print(rep.summary())
raw, out, face = samples[-1]
cv2.imwrite("/tmp/bedhead_bench/final_raw.png", raw)
cv2.imwrite("/tmp/bedhead_bench/final_out.png", out)
