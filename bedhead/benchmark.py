"""bedhead quality benchmark: 8 metrics that define 'outperform the market'.

Each metric is measured on live captures against a reference photo.
Targets were authored from public competitor claims and spec sheets (what
a good retouch product should deliver); competitors were NOT run locally,
so these are engineering targets, not head-to-head measurements.
Output: PASS/FAIL per target + a composite score. This is the acceptance
gate for the quality bar -- no subjective claims.

Metrics:
  1. under_eye_gap    cheek-to-undereye L* deficit after correction
                      (competitor-class: < 12 levels)          target: < 10
  2. skin_texture_hf  high-frequency detail retention in face region
                      (waxy = fail; meitu-class keeps >= 85%)  target: >= 90% of raw
  3. color_fidelity   face LAB delta vs reference look         target: dE < 8
  4. exposure_gap     face L* vs reference L*                  target: |gap| < 8
  5. subject_pop      face-to-background L* separation         target: >= 15
  6. saturation       face sat vs reference sat                target: within +-10
  7. temporal_stability  frame-to-frame face warmth/grade pumping target: < 2.0/frame
  8. latency          end-to-end ms at 720p (Broadcast GPU: ~16-33ms; our CPU
                      budget: < 45 ms so 24fps+ holds)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import cv2
import numpy as np

from .guard import IdentityGuard
from .retoucher import under_eye_mask
from .tracker import FACE_OVAL, FaceFrame


@dataclass
class MetricResult:
    name: str
    value: float
    target: float
    passed: bool
    detail: str = ""


@dataclass
class BenchReport:
    results: list[MetricResult] = field(default_factory=list)
    passed: int = 0
    total: int = 0

    def add(self, name: str, value: float, target: float, passed: bool, detail: str = "") -> None:
        self.results.append(MetricResult(name, round(value, 2), target, passed, detail))
        self.total += 1
        self.passed += int(passed)

    def summary(self) -> str:
        lines = [f"{'metric':24s} {'value':>8} {'target':>8}  verdict"]
        for r in self.results:
            lines.append(f"{r.name:24s} {r.value:8.2f} {r.target:8.2f}  {'PASS' if r.passed else 'FAIL'}"
                         + (f"  ({r.detail})" if r.detail else ""))
        lines.append(f"\nSCORE: {self.passed}/{self.total} targets met")
        return "\n".join(lines)


def _face_pts(face: FaceFrame):
    return face.landmarks[list(FACE_OVAL), :2].astype(int)


def _oval_mask(face: FaceFrame, shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.float32)
    cv2.fillPoly(m, [_face_pts(face)], 1.0)
    return m


def _band_mask(face: FaceFrame, shape) -> np.ndarray:
    return under_eye_mask(face)


def _bg_mask(face: FaceFrame, shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.float32)
    cv2.rectangle(m, (0, 0), (shape[1] - 1, shape[0] - 1), 1.0, -1)
    m = m - _oval_mask(face, shape)
    return np.clip(m, 0, 1)


def _hf_energy(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    if mask is not None:
        sel = hf[mask > 0.5]
        return float(sel.std()) if sel.size else 0.0
    return float(hf.std())


def _ref_face_mask(reference: np.ndarray) -> np.ndarray | None:
    """Detect the face in the reference photo; None if no face found."""
    try:
        g = IdentityGuard()
        faces = g._app.get(reference)
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        h, w = reference.shape[:2]
        m = np.zeros((h, w), np.float32)
        x0, y0, x1, y1 = [int(v) for v in f.bbox]
        # tighten to the central 70% (bbox includes hair/ears; we want skin)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        bw, bh = int((x1 - x0) * 0.35), int((y1 - y0) * 0.35)
        cv2.ellipse(m, (cx, cy), (bw, bh), 0, 0, 360, 1.0, -1)
        return m
    except Exception:  # noqa: BLE001
        return None


def _lab(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def benchmark(
    raw: np.ndarray,
    out: np.ndarray,
    reference: np.ndarray,
    face: FaceFrame,
    latency_ms: float | None = None,
    temporal_deltas: list[float] | None = None,
) -> BenchReport:
    """Score one processed frame (with its raw source) against the targets."""
    rep = BenchReport()
    oval = _oval_mask(face, raw.shape)
    band = _band_mask(face, raw.shape)
    bg = _bg_mask(face, raw.shape)

    _lab_raw, lab_out, lab_ref = _lab(raw), _lab(out), _lab(reference)
    hsv_out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)

    # 1. under-eye gap (band vs cheek=oval complement above band)
    cheek = lab_out[..., 0][oval > 0.5]
    band_l = lab_out[..., 0][band > 0.5]
    cheek_p60 = float(np.percentile(cheek, 60)) if cheek.size else 0
    band_p10 = float(np.percentile(band_l, 10)) if band_l.size else 0
    gap = cheek_p60 - band_p10
    rep.add("under_eye_gap", gap, 10.0, gap < 10.0,
            f"cheek p60 {cheek_p60:.0f} vs band p10 {band_p10:.0f}")

    # 2. skin texture retention
    hf_raw = _hf_energy(raw, oval)
    hf_out = _hf_energy(out, oval)
    ratio = hf_out / max(hf_raw, 1e-6)
    rep.add("skin_texture_hf_pct", ratio * 100, 90.0, ratio >= 0.90,
            f"raw {hf_raw:.1f} -> out {hf_out:.1f}")

    # 3-6: compare against the reference's ACTUAL face region (detect it)
    ref_detect = _ref_face_mask(reference)
    rm = ref_detect if ref_detect is not None else None
    if rm is None:
        # fallback: center crop (assume portrait single-subject selfie)
        rm = np.zeros(reference.shape[:2], np.float32)
        h_r, w_r = reference.shape[:2]
        cv2.ellipse(rm, (w_r // 2, h_r // 2), (w_r // 4, h_r // 3), 0, 0, 360, 1.0, -1)
    dE_a = abs(float(lab_out[..., 1][oval > 0.5].mean()) - float(lab_ref[..., 1][rm > 0.5].mean()))
    dE_b = abs(float(lab_out[..., 2][oval > 0.5].mean()) - float(lab_ref[..., 2][rm > 0.5].mean()))
    rep.add("color_fidelity_dE", (dE_a**2 + dE_b**2) ** 0.5, 8.0,
            (dE_a**2 + dE_b**2) ** 0.5 < 8.0, f"da {dE_a:.1f} db {dE_b:.1f}")

    # 4. exposure
    gapL = float(lab_out[..., 0][oval > 0.5].mean()) - float(lab_ref[..., 0][rm > 0.5].mean())
    rep.add("exposure_gap", abs(gapL), 8.0, abs(gapL) < 8.0, f"L* out {gapL:+.1f} vs ref")

    # 5. subject pop: face vs background L* separation
    face_l = float(lab_out[..., 0][oval > 0.5].mean())
    bg_l = float(lab_out[..., 0][bg > 0.5].mean())
    pop = face_l - bg_l
    rep.add("subject_pop", pop, 15.0, pop >= 15.0, f"face {face_l:.0f} vs bg {bg_l:.0f}")

    # 6. saturation
    sat_out = float(hsv_out[..., 1][oval > 0.5].mean())
    sat_ref = float(hsv_ref[..., 1][rm > 0.5].mean())
    rep.add("saturation_delta", abs(sat_out - sat_ref), 10.0,
            abs(sat_out - sat_ref) < 10.0, f"out {sat_out:.0f} vs ref {sat_ref:.0f}")

    # 7. temporal stability (needs >= 2 deltas from the caller)
    if temporal_deltas:
        pump = statistics.mean(temporal_deltas)
        rep.add("temporal_pump", pump, 2.0, pump < 2.0, f"mean frame warmth swing {pump:.2f}")

    # 8. latency
    if latency_ms is not None:
        rep.add("latency_ms", latency_ms, 45.0, latency_ms < 45.0,
                f"{latency_ms:.0f} ms -> {1000 / max(latency_ms, 1e-6):.0f} fps")

    return rep
