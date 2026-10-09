"""Reference-guided autotune: derive effect strengths from a reference photo.

Compares a live frame against the admitted reference photo and computes the
Tier A adjustments that move the live look toward the reference look:

  - exposure/warmth gap  -> soft_light + studio_light strengths
  - skin-tone gap        -> under_eye / shine strengths (bounded)
  - sharpness gap        -> detail/sharpen strength

All outputs are 0..1 strengths for the existing Preset fields, clamped, and
mergeable over the user's current preset. Nothing generative: this tunes
classical effects only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .tracker import FACE_OVAL, FaceFrame


@dataclass
class AutotuneResult:
    preset_delta: dict[str, float]   # field -> suggested value (not delta)
    notes: list[str]


def _stats(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Robust per-channel stats (median) + luminance + saturation + sharpness."""
    sel = None
    if mask is not None:
        m = (mask > 0.5).astype(np.uint8)
        if m.sum() >= 100:
            sel = img_bgr[m.astype(bool)]
    if sel is None:
        sel = img_bgr.reshape(-1, 3)
    med = np.median(sel, axis=0).astype(np.float32)  # BGR
    lab = cv2.cvtColor(med.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
    L = float(lab[0, 0, 0])
    a = float(lab[0, 0, 1])   # green(-) .. red(+)
    b = float(lab[0, 0, 2])   # blue(-) .. yellow(+)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if mask is not None and sel is not img_bgr.reshape(-1, 3):
        mm = (mask > 0.5)
        sat = float(hsv[..., 1][mm[:hsv.shape[0], :hsv.shape[1]]].mean())
    else:
        sat = float(hsv[..., 1].mean())
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"med_bgr": med, "L": L, "a": a, "b": b, "sat": sat, "sharp": sharp}


def _clamp01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _band_gap(live_bgr: np.ndarray, face: FaceFrame) -> float | None:
    """Cheek-p60 minus under-eye-band-p10 L* on the live frame (None if n/a)."""
    try:
        from .retoucher import under_eye_mask

        band = under_eye_mask(face)
        if (band > 100).sum() < 200:
            return None
        lab_live = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        band_p10 = float(np.percentile(lab_live[..., 0][band > 100], 10))
        om = np.zeros(live_bgr.shape[:2], np.uint8)
        cv2.fillPoly(om, [face.landmarks[list(FACE_OVAL), :2].astype(np.int32)], 255)
        cheek_p60 = float(np.percentile(lab_live[..., 0][om > 0], 60))
        return cheek_p60 - band_p10
    except Exception:  # noqa: BLE001
        return None


def autotune(
    live_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    live_face_mask: np.ndarray | None = None,
    reference_face_mask: np.ndarray | None = None,
    face: FaceFrame | None = None,
) -> AutotuneResult:
    """Suggest preset strengths that move the live frame toward the reference."""
    s_live = _stats(live_bgr, live_face_mask)
    s_ref = _stats(reference_bgr, reference_face_mask)
    notes: list[str] = []

    dL = s_ref["L"] - s_live["L"]          # + means reference brighter
    d_warm = s_ref["b"] - s_live["b"]      # + means reference warmer/yellower
    d_sharp = s_ref["sharp"] - s_live["sharp"]

    delta: dict[str, float] = {}

    # Exposure: soft_light lifts up to ~+9 R; scale by how dark we are vs ref.
    if dL > 4:
        delta["soft_light"] = _clamp01(dL / 20.0)  # measured: strength 1.0 lifts face L* ~20
        notes.append(f"live is {dL:.0f} L* darker than reference -> soft_light {delta['soft_light']:.2f}")
    elif dL < -4:
        delta["soft_light"] = 0.0
        notes.append(f"live is {-dL:.0f} L* brighter than reference -> soft_light off")
    else:
        delta["soft_light"] = 0.1

    # Warmth: soft_light adds warmth; studio_light adds person-only warmth.
    if d_warm > 3:
        delta["studio_light"] = _clamp01(d_warm / 30.0)
        notes.append(f"reference is warmer (db* {d_warm:.0f}) -> studio_light {delta['studio_light']:.2f}")
    else:
        delta["studio_light"] = 0.0

    # Vibrance: webcams desaturate; lift dull pixels toward the reference sat.
    d_sat = s_ref.get("sat", 0.0) - s_live.get("sat", 0.0)
    if d_sat > 5:
        delta["vibrance"] = _clamp01(d_sat / 40.0)
        notes.append(f"live face desaturated (dsat {d_sat:.0f}) -> vibrance {delta['vibrance']:.2f}")
    else:
        delta["vibrance"] = 0.0

    # Under-eye: measure the ACTUAL band deficit on the live frame via the
    # landmark mask (was a blind 0.4 default; bench showed 0.4 leaves a
    # 30-level gap when the real deficit needs ~0.9)
    gap = _band_gap(live_bgr, face) if face is not None else None
    if gap is not None:
        # gap 30+ -> full 0.9; gap <= 8 -> 0
        delta["under_eye"] = _clamp01((gap - 8.0) / 24.0)
        notes.append(f"under-eye gap {gap:.0f} -> under_eye {delta['under_eye']:.2f}")
    elif dL > 8:
        delta["under_eye"] = _clamp01(0.3 + dL / 80.0)
        notes.append("reference notably brighter -> under_eye lift suggested")
    else:
        delta["under_eye"] = _clamp01(0.3 + dL / 80.0)

    # Sharpness: reference sharper (typical for a good photo) -> add detail.
    if d_sharp > 20:
        ratio = s_ref["sharp"] / max(s_live["sharp"], 1.0)
        delta["skin"] = _clamp01(0.4 + min(ratio - 1.0, 1.0) * 0.25)
        # reference genuinely sharper -> sharpen the live face toward it
        delta["detail_boost"] = _clamp01(min(ratio - 1.0, 1.5) * 0.4)
        notes.append(f"reference sharper (x{ratio:.1f}) -> skin smoothing reduced, detail_boost {delta['detail_boost']:.2f}")
    else:
        delta["skin"] = 0.5
        delta["detail_boost"] = 0.0

    delta["shine"] = 0.5
    delta["intensity"] = 0.7
    return AutotuneResult(preset_delta=delta, notes=notes)


def apply_autotune(base_preset_dict: dict, result: AutotuneResult) -> dict:
    """Merge autotune suggestions onto a preset dict (autotune wins)."""
    merged = dict(base_preset_dict)
    merged.update(result.preset_delta)
    return merged


# ---------------------------------------------------------------------
# Continuous mode: re-measure periodically while the call runs.
# ---------------------------------------------------------------------

# Fields the continuous tracker may adapt (ambient-driven, never taste keys).
CONTINUOUS_FIELDS = ("soft_light", "studio_light")


class LookTracker:
    """Tracks the live look vs the reference and adapts lighting strengths.

    Runs a measurement + suggest cycle at most every `interval` seconds and
    EMA-smooths the suggested strengths so changes never pop. Only
    CONTINUOUS_FIELDS are touched; user taste keys (skin, teeth, ...) stay
    wherever the user set them. A deadband suppresses micro-adjustments.
    """

    def __init__(
        self,
        reference_bgr: np.ndarray,
        interval_s: float = 2.0,
        ema: float = 0.25,
        deadband: float = 0.03,
        face_mask_fn=None,
        face_fn=None,  # () -> FaceFrame | None, for under-eye band measurement
    ) -> None:
        self._ref_stats = _stats(reference_bgr)
        self.interval = interval_s
        self.ema = ema            # weight of the NEW suggestion per cycle
        self.deadband = deadband  # ignore suggested deltas below this
        self._face_mask_fn = face_mask_fn
        self._face_fn = face_fn
        self._last_run: float | None = None  # clock-agnostic; set on first tick
        self.current: dict[str, float] = {k: 0.0 for k in CONTINUOUS_FIELDS}
        self._primed = False
        self.notes: list[str] = []

    def prime(self, frame_bgr: np.ndarray) -> None:
        """Seed strengths from one frame (the startup measurement)."""
        at = self._suggest(frame_bgr)
        for k in CONTINUOUS_FIELDS:
            self.current[k] = at.get(k, 0.0)
        self._primed = True
        self._last_run = None  # next tick re-syncs the clock

    def tick(self, frame_bgr: np.ndarray, now: float | None = None) -> bool:
        """Maybe re-measure; returns True if strengths changed."""
        if not self._primed:
            self.prime(frame_bgr)
            return True
        t = time.monotonic() if now is None else now
        if self._last_run is None:
            self._last_run = t  # first tick after prime: sync clock, no measure
            return False
        if t - self._last_run < self.interval:
            return False
        self._last_run = t
        at = self._suggest(frame_bgr)
        changed = False
        self.notes = []
        for k in CONTINUOUS_FIELDS:
            target = at.get(k, 0.0)
            if abs(target - self.current[k]) < self.deadband:
                continue
            self.current[k] = (1.0 - self.ema) * self.current[k] + self.ema * target
            changed = True
        return changed

    def _suggest(self, frame_bgr: np.ndarray) -> dict[str, float]:
        mask = self._face_mask_fn() if self._face_mask_fn is not None else None
        face = self._face_fn() if self._face_fn is not None else None
        s_live = _stats(frame_bgr, mask)
        dL = self._ref_stats["L"] - s_live["L"]
        d_warm = self._ref_stats["b"] - s_live["b"]
        out: dict[str, float] = {}
        # exposure: suggestion tracks the live gap continuously; inside the
        # +-4 deadband the gap is closed, so ease off (never "hold": a held
        # high strength would over-brighten once ambient recovers)
        out["soft_light"] = _clamp01(dL / 20.0)
        if dL < 0:
            out["soft_light"] = 0.0
        if d_warm > 3:
            out["studio_light"] = _clamp01(d_warm / 30.0)
        else:
            out["studio_light"] = 0.0
        # under-eye: measured band deficit, same rule as the one-shot path
        if face is not None:
            gap = _band_gap(frame_bgr, face)
            if gap is not None:
                out["under_eye"] = _clamp01((gap - 8.0) / 24.0)
        return out
