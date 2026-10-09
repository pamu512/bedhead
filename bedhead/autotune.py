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

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AutotuneResult:
    preset_delta: dict[str, float]   # field -> suggested value (not delta)
    notes: list[str]


def _stats(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Robust per-channel stats (median) + luminance + sharpness (Laplacian var)."""
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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"med_bgr": med, "L": L, "a": a, "b": b, "sharp": sharp}


def _clamp01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def autotune(
    live_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    live_face_mask: np.ndarray | None = None,
    reference_face_mask: np.ndarray | None = None,
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
        delta["soft_light"] = _clamp01(dL / 40.0)
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

    # Under-eye: darker under-eye area in live vs ref is hard to measure
    # without landmarks; keep bounded default when ref looks fresher (brighter L).
    if dL > 8:
        delta["under_eye"] = _clamp01(0.3 + dL / 80.0)
        notes.append("reference notably brighter -> under_eye lift suggested")
    else:
        delta["under_eye"] = 0.4

    # Sharpness: reference sharper (typical for a good photo) -> add detail.
    if d_sharp > 20:
        ratio = s_ref["sharp"] / max(s_live["sharp"], 1.0)
        delta["skin"] = _clamp01(0.4 + min(ratio - 1.0, 1.0) * 0.25)
        notes.append(f"reference sharper (x{ratio:.1f}) -> skin smoothing reduced, edges preserved")
    else:
        delta["skin"] = 0.5

    delta["shine"] = 0.5
    delta["intensity"] = 0.7
    return AutotuneResult(preset_delta=delta, notes=notes)


def apply_autotune(base_preset_dict: dict, result: AutotuneResult) -> dict:
    """Merge autotune suggestions onto a preset dict (autotune wins)."""
    merged = dict(base_preset_dict)
    merged.update(result.preset_delta)
    return merged
