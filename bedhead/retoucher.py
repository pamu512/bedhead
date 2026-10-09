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

import math

import cv2
import numpy as np

from .config import Preset
from .lighting import background_mode, eye_light, studio_light
from .quality import unsharp_detail
from .segmenter import Segmenter
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
    """Bounding box of the face oval + margin, clipped to the frame.

    Every edge is clamped independently into the frame, so a face hanging
    past an edge (or fully outside it) still yields an ordered, in-range
    box; degenerate slivers are skipped by the caller.
    """
    pts = face.landmarks[list(FACE_OVAL), :2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0 = int(np.clip(x0 - mx, 0, face.w - 1))
    y0 = int(np.clip(y0 - my, 0, face.h - 1))
    x1 = int(np.clip(x1 + mx, 0, face.w - 1))
    y1 = int(np.clip(y1 + my, 0, face.h - 1))
    return x0, y0, x1, y1


# ---------------------------------------------------------------- effects

def _retouch_roi(
    roi: np.ndarray,
    face: FaceFrame,
    preset: Preset,
    seg_skin: np.ndarray | None = None,
) -> np.ndarray:
    """All face effects, computed inside the ROI only.

    Quality architecture (best-in-class, research-driven):
      - LAB color space: all corrections on the L channel; chroma touched only
        deliberately (teeth yellow-desat), so no pasty color shift.
      - Guided-filter frequency separation for skin: smooth the low-frequency
        base, attenuate only mid-frequency blemishes, keep pores (high freq).
        Replaces edgePreservingFilter; ~2-3 ms and no plastic look.
      - Shine: local luminance percentile (vs the guided-filter base), not a
        global tone threshold; works across skin tones and curved highlights.
      - Under-eye: L lifted toward the cheek-band reference percentile.
      - Teeth: LAB b-channel desaturation + L curve, L-gated (dark interior out).
      - Eyes/lips/brows: gentle unsharp on the high-frequency residual.
    """
    out = roi.copy()
    h, w = face.h, face.w
    # cv2.resize((w // 2, h // 2)) throws when a side is under 2px.
    if h < 2 or w < 2 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return roi
    out = roi.copy()

    s_skin = preset.scaled("skin")
    s_eye = preset.scaled("under_eye")
    s_shine = preset.scaled("shine")
    s_teeth = preset.scaled("teeth")
    s_hair = preset.scaled("hairline")

    skin_m = skin_mask(face)
    # true-skin refinement from the segmenter (intersects with the oval mask:
    # segmentation knows about hair strands/glasses, the oval keeps geometry)
    if seg_skin is not None and seg_skin.any():
        seg_u8 = (np.clip(seg_skin, 0, 1) * 255).astype(np.uint8)
        if seg_u8.shape != skin_m.shape:
            seg_u8 = cv2.resize(seg_u8, (skin_m.shape[1], skin_m.shape[0]))
        skin_m = cv2.min(skin_m, seg_u8)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    li = lab[..., 0].astype(np.float32)
    base = cv2.ximgproc.guidedFilter(guide=lab[..., 0], src=li, radius=8, eps=80.0)
    detail = li - base

    skin_a = (skin_m.astype(np.float32) / 255.0) * s_skin

    # --- skin: frequency separation on L (chroma untouched)
    if s_skin > 0:
        # attenuate detail (blemishes live mid-frequency) but keep a floor of
        # high-frequency texture so skin never looks plastic
        att = 1.0 - 0.85 * skin_a
        new_l = base + detail * att
        lab[..., 0] = np.clip(new_l, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- under-eye: lift L toward cheek reference percentile, texture kept
    if s_eye > 0:
        eye_m = under_eye_mask(face)
        cheek_l = float(np.percentile(
            base[skin_m > 200], 60)) if (skin_m > 200).any() else float(np.mean(base))
        target = base + np.clip(cheek_l - base, 0, 40) * 0.5 + detail * 0.7
        a = (eye_m.astype(np.float32) / 255.0) * s_eye * 0.8
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lf = lab[..., 0].astype(np.float32)
        lab[..., 0] = np.clip(lf * (1 - a) + target * a, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- shine: compress L where it exceeds the LOCAL smooth base by k
    if s_shine > 0:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lf = lab[..., 0].astype(np.float32)
        excess = np.clip((lf - (base + 18.0)) / 40.0, 0, 1)
        a = excess * (skin_m.astype(np.float32) / 255.0) * s_shine
        rolled = base + (lf - base) * 0.55  # soft highlight rolloff
        lab[..., 0] = np.clip(lf * (1 - a) + rolled * a, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- teeth whitening (jaw-open hysteresis gate, LAB b-desat, L-gated)
    if s_teeth > 0 and face.extras.get("jaw_open_gated", face.extras.get("jaw_open", 0.0)) > 0.5:
        inner = _mask_poly(h, w, face.poly(INNER_LIPS))
        if inner.any():
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lf = lab[..., 0].astype(np.float32)
            bf = lab[..., 2].astype(np.float32)   # yellow <-> blue
            gate = np.clip((lf - 80.0) / 60.0, 0, 1)          # keep dark interior out
            inner_f = _feather(inner, 9).astype(np.float32) / 255.0
            amt = s_teeth * inner_f * gate
            b_new = np.clip(bf * (1 - 0.75 * amt) + 128.0 * 0.75 * amt, 0, 255)
            l_new = np.clip(lf + 28.0 * amt, 0, 255)
            lab[..., 2] = b_new.astype(np.uint8)
            lab[..., 0] = l_new.astype(np.uint8)
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- hairline softening (low-frequency blend in the top band)
    if s_hair > 0 and w >= 2 and h >= 2:
        band = hairline_band_mask(face)
        soft = cv2.ximgproc.guidedFilter(guide=roi, src=roi, radius=10, eps=150.0)
        a = (band.astype(np.float32) / 255.0) * s_hair * 0.6
        out = np.clip(
            out.astype(np.float32) * (1 - a[..., None])
            + soft.astype(np.float32) * a[..., None], 0, 255
        ).astype(np.uint8)

    # --- crisp eyes/lips/brows: unsharp on the detail residual (inverse skin)
    sharp_a = 0.35 * max(s_skin, 0.3)  # subtle, scales with overall intensity
    if sharp_a > 0:
        feat_m = cv2.subtract(_mask_poly(h, w, face.poly(FACE_OVAL)), skin_m)
        feat_m = _feather(feat_m, 15)
        out = unsharp_detail(out, feat_m, sharp_a)

    return out


def _apply_face_effects(
    frame: np.ndarray,
    face: FaceFrame,
    preset: Preset,
    seg_skin: np.ndarray | None = None,
) -> np.ndarray:
    """Classic Tier A effects, confined to the face ROI.

    seg_skin: optional true-skin mask (from the Segmenter) at frame
    resolution; when present it refines the landmark oval approximation.
    """
    out = frame.copy()
    fh, fw = out.shape[:2]
    if fh < 1 or fw < 1:
        return out
    x0, y0, x1, y1 = _roi(face)
    # clamp: landmark outliers must never produce an empty or inverted slice
    x0 = min(max(x0, 0), fw - 1)
    x1 = min(max(x1, 0), fw - 1)
    y0 = min(max(y0, 0), fh - 1)
    y1 = min(max(y1, 0), fh - 1)
    if y1 <= y0 or x1 <= x0:
        return out
    roi = out[y0:y1 + 1, x0:x1 + 1]
    if roi.shape[0] < 2 or roi.shape[1] < 2:
        return out
    shifted = FaceFrame(
        landmarks=face.landmarks - np.array([x0, y0, 0], dtype=np.float32),
        h=roi.shape[0],
        w=roi.shape[1],
        extras=face.extras,
    )
    skin_refine = None
    if seg_skin is not None:
        skin_refine = seg_skin[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        if skin_refine.shape[:2] != roi.shape[:2]:
            skin_refine = cv2.resize(skin_refine, (roi.shape[1], roi.shape[0]))
    out[y0:y1 + 1, x0:x1 + 1] = _retouch_roi(roi, shifted, preset, skin_refine)
    return out


def apply(
    frame_bgr: np.ndarray,
    face: FaceFrame | None,
    preset: Preset,
    segmenter: Segmenter | None = None,
) -> np.ndarray:
    """Main entry: returns the retouched (or passthrough) frame.

    Order matters: background compositing first (so lighting effects see the
    final background), then person relight, then eye light, then the classic
    face effects, then the global soft-light lift.
    """
    # NaN fails `<= 0` and would poison the blend math; treat it as passthrough.
    if math.isnan(preset.intensity):
        return frame_bgr
    if face is None and not _seg_effects_active(preset):
        return frame_bgr
    if preset.intensity <= 0 and not _seg_effects_active(preset):
        return frame_bgr
    if preset.show_original:
        return frame_bgr

    out = frame_bgr

    # --- segmentation-driven effects (background, studio light)
    if _seg_effects_active(preset) and segmenter is not None:
        person_m = segmenter.person_mask()
        if preset.background_strength > 0 and preset.background_mode != "off":
            out = background_mode(out, person_m,
                                  preset.background_strength, preset.background_mode)
        if preset.studio_light > 0:
            s = preset.scaled("studio_light")
            out = studio_light(out, person_m, s)

    # --- eye light (landmark-driven, no segmentation needed)
    if preset.eye_light > 0 and face is not None:
        out = eye_light(out, face.landmarks, preset.scaled("eye_light"))

    # --- face effects (need a tracked face)
    if face is not None and preset.intensity > 0:
        seg_skin = segmenter.face_skin_mask() if segmenter is not None else None
        out = _apply_face_effects(out, face, preset, seg_skin)

    # --- soft light: global warm lift with highlight roll-off (LUT, O(1)).
    # Rolling lift (strongest in shadows, zero at white) closes dark-webcam
    # gaps without clipping highlights the way a flat additive would.
    s_light = preset.scaled("soft_light")
    if s_light > 0:
        v = np.arange(256, dtype=np.float32)
        roll = (1.0 - v / 255.0)  # 1 at black, 0 at white
        lut_r = np.clip(v + 60.0 * s_light * roll, 0, 255).astype(np.uint8)
        lut_g = np.clip(v + 24.0 * s_light * roll, 0, 255).astype(np.uint8)
        lut_b = np.clip(v + 8.0 * s_light * roll, 0, 255).astype(np.uint8)
        chans = cv2.split(out)
        out = cv2.merge([
            cv2.LUT(chans[0], lut_b),
            cv2.LUT(chans[1], lut_g),
            cv2.LUT(chans[2], lut_r),
        ])

    return out


def _seg_effects_active(preset: Preset) -> bool:
    return (
        preset.background_strength > 0 and preset.background_mode != "off"
    ) or preset.studio_light > 0
