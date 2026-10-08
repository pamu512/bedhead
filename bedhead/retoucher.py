"""Tier A retoucher: mesh-aware corrective makeup, classical CV only.

Effects (all landmark-driven, feathered, capped so nothing can run away):
  skin       - edge-preserving smoothing restricted to skin polygon
  under_eye  - brightness/alpha lift in the lower-orbit band
  shine      - compress highlights on skin that exceed the face tone
  teeth      - desaturate + lift inside inner lips when jaw is open
  hairline   - EXPERIMENTAL band softening at the top oval edge
  soft_light - global gentle exposure + warmth lift (NVIDIA-brightness vibe)

Design rules:
  - every effect is gated by Preset.scaled() (0 disables the code path)
  - every regional edit is feathered and clamped so seams never pop
  - failing tracking => frame passes through untouched (fail-safe to reality)
  - all face effects run inside the face ROI (3-4x cheaper than full frame)
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Preset
from .tracker import (
    CHEEK_SAMPLES,
    FACE_OVAL,
    FOREHEAD_SAMPLES,
    INNER_LIPS,
    LEFT_BROW,
    LEFT_EYE_RING,
    OUTER_LIPS,
    RIGHT_BROW,
    RIGHT_EYE_RING,
    FaceFrame,
)

# ---------------------------------------------------------------- masks

def _mask_poly(h: int, w: int, polys: tuple[np.ndarray, ...] | np.ndarray) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, polys if isinstance(polys, tuple) else (polys,), 255)
    return m


def _feather(mask: np.ndarray, k: int) -> np.ndarray:
    k = k | 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def skin_mask(face: FaceFrame) -> np.ndarray:
    """Face oval minus eyes/brows/lips, feathered."""
    h, w = face.h, face.w
    skin = _mask_poly(h, w, face.poly(FACE_OVAL))
    holes = [
        _mask_poly(h, w, face.poly(LEFT_EYE_RING)),
        _mask_poly(h, w, face.poly(RIGHT_EYE_RING)),
        _mask_poly(h, w, face.poly(LEFT_BROW)),
        _mask_poly(h, w, face.poly(RIGHT_BROW)),
        _mask_poly(h, w, face.poly(OUTER_LIPS)),
    ]
    for hm in holes:
        skin = cv2.subtract(skin, hm)
    return _feather(skin, 25)


def under_eye_mask(face: FaceFrame) -> np.ndarray:
    """Band between lower eye ring and cheek: offset lower-eyelid rim points downward."""
    h, w = face.h, face.w
    left_lower = (145, 153, 154, 155, 133, 172, 157, 158, 159, 160, 161)
    right_lower = (374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388)
    band = np.zeros((h, w), np.uint8)
    for rim in (left_lower, right_lower):
        pts = face.landmarks[list(rim), :2].copy()
        span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) + 1e-6
        drop = max(2.0, span * 0.06)  # band height ~6% of eye width
        pts[:, 1] += drop
        poly = np.round(pts).astype(np.int32)
        cv2.fillPoly(band, [poly], 255)
    band = cv2.subtract(band, _mask_poly(h, w, face.poly(LEFT_EYE_RING)))
    band = cv2.subtract(band, _mask_poly(h, w, face.poly(RIGHT_EYE_RING)))
    band = cv2.subtract(band, _mask_poly(h, w, face.poly(OUTER_LIPS)))
    return _feather(band, 21)


def hairline_band_mask(face: FaceFrame) -> np.ndarray:
    """Top slice of the face oval: between forehead samples and the oval top edge."""
    h, w = face.h, face.w
    oval = _mask_poly(h, w, face.poly(FACE_OVAL))
    c = face.landmarks[list(FOREHEAD_SAMPLES), :2].mean(axis=0)
    cut = int(c[1] + (c[1] * 0.08))
    top = np.zeros_like(oval)
    top[: max(cut, 1), :] = 255
    band = cv2.bitwise_and(oval, top)
    return _feather(band, 31)


def _skin_tone_stats(roi_bgr: np.ndarray, face: FaceFrame) -> tuple[np.ndarray, float]:
    """Median tone + brightness of sample skin pixels (robust to bed-head days)."""
    pts = np.round(face.landmarks[list(CHEEK_SAMPLES), :2]).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, face.w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, face.h - 1)
    vals = roi_bgr[pts[:, 1], pts[:, 0]].astype(np.float32)
    tone = np.median(vals, axis=0)
    bright = float(np.percentile(cv2.cvtColor(vals.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY), 75))
    return tone, bright


def _roi(face: FaceFrame, margin: float = 0.18) -> tuple[int, int, int, int]:
    """Bounding box of the face oval + margin, clipped to the frame."""
    pts = face.landmarks[list(FACE_OVAL), :2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0 = int(max(0, x0 - mx))
    y0 = int(max(0, y0 - my))
    x1 = int(min(face.w - 1, x1 + mx))
    y1 = int(min(face.h - 1, y1 + my))
    return x0, y0, x1, y1


# ---------------------------------------------------------------- effects

def _retouch_roi(roi: np.ndarray, face: FaceFrame, preset: Preset) -> np.ndarray:
    """All face effects, computed inside the ROI only."""
    out = roi.copy()
    h, w = face.h, face.w

    s_skin = preset.scaled("skin")
    s_eye = preset.scaled("under_eye")
    s_shine = preset.scaled("shine")
    s_teeth = preset.scaled("teeth")
    s_hair = preset.scaled("hairline")

    skin_m = skin_mask(face)
    tone, tone_bright = _skin_tone_stats(roi, face)

    # --- skin smoothing (edge-preserving, skin-only, computed at half res)
    if s_skin > 0:
        small = cv2.resize(roi, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        smooth_small = cv2.edgePreservingFilter(small, flags=1, sigma_s=60, sigma_r=0.45)
        smooth = cv2.resize(smooth_small, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = (skin_m.astype(np.float32) / 255.0) * s_skin
        out = np.clip(
            out.astype(np.float32) * (1 - alpha[..., None]) + smooth.astype(np.float32) * alpha[..., None],
            0, 255,
        ).astype(np.uint8)

    # --- under-eye brighten (screen-style, capped)
    if s_eye > 0:
        eye_m = under_eye_mask(face)
        target = np.clip(tone * 1.25 + 18, 0, 255)
        alpha = (eye_m.astype(np.float32) / 255.0) * s_eye * 0.85
        out = np.clip(
            out * (1 - alpha[..., None]) + target * alpha[..., None], 0, 255
        ).astype(np.uint8)

    # --- shine control (compress specular highlights on skin)
    if s_shine > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hot = np.clip((gray - (tone_bright + 35)) / 60.0, 0, 1)
        hot = hot * (skin_m.astype(np.float32) / 255.0) * s_shine
        suppressed = np.minimum(out.astype(np.float32), tone * 1.15 + 30)
        out = np.clip(out * (1 - hot[..., None]) + suppressed * hot[..., None], 0, 255).astype(np.uint8)

    # --- teeth whitening (mouth-open gated by blendshape)
    if s_teeth > 0 and face.extras.get("jaw_open", 0.0) > 0.25:
        inner = _mask_poly(h, w, face.poly(INNER_LIPS))
        if inner.any():
            inner_f = _feather(inner, 9).astype(np.float32) / 255.0
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] - 55 * s_teeth * inner_f, 0, 255).astype(np.uint8)
            hsv[..., 2] = np.clip(hsv[..., 2] + 45 * s_teeth * inner_f, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- hairline softening (experimental, half-res blur)
    if s_hair > 0:
        band = hairline_band_mask(face)
        small = cv2.resize(roi, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        blur_small = cv2.bilateralFilter(small, 9, 50, 50)
        blur = cv2.resize(blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = (band.astype(np.float32) / 255.0) * s_hair * 0.6
        out = np.clip(
            out * (1 - alpha[..., None]) + blur.astype(np.float32) * alpha[..., None], 0, 255
        ).astype(np.uint8)

    return out


def apply(
    frame_bgr: np.ndarray,
    face: FaceFrame | None,
    preset: Preset,
    cache: dict | None = None,
) -> np.ndarray:
    """Main entry: returns the retouched (or passthrough) frame."""
    if cache is None:
        cache = {}
    if face is None or preset.intensity <= 0:
        return frame_bgr
    if preset.show_original:
        return frame_bgr

    out = frame_bgr.copy()

    # --- soft light: global gentle warm lift (cheap saturated add, no float pass)
    s_light = preset.scaled("soft_light")
    if s_light > 0:
        lift = (0, 4 * s_light, 9 * s_light)  # BGR: warm = +R
        out = cv2.add(out, tuple(int(round(v)) for v in lift))

    # --- face effects inside the ROI only
    x0, y0, x1, y1 = _roi(face)
    roi = out[y0:y1 + 1, x0:x1 + 1]
    shifted = FaceFrame(
        landmarks=face.landmarks - np.array([x0, y0, 0], dtype=np.float32),
        score=face.score,
        h=roi.shape[0],
        w=roi.shape[1],
        extras=face.extras,
    )
    out[y0:y1 + 1, x0:x1 + 1] = _retouch_roi(roi, shifted, preset)
    return out
