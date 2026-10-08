"""Clothes tidy-up: "freshly steamed" shirt, classical CV on a clothes mask.

Uses MediaPipe Selfie Multiclass segmentation (model already shipped in
models.py) to isolate the garment, then two bounded effects:

  clothes - crease/wrinkle softening: edge-preserving smoothing inside the
            garment mask flattens wrinkle shading while the garment silhouette
            and strong features (collar, buttons) survive
  stain   - color-unevenness pull: pixels that drift a low-to-moderate amount
            from the garment's median color are nudged back (faint stains,
            shadowing). Extreme deviations (crisp prints, stripes) are left
            alone - this is a steam iron, not a repaint.

Design rules (same as the face retoucher):
  - effects gated by Preset.scaled() (0 disables the code path)
  - mask feathered, alphas capped, everything confined to the mask ROI
  - segmentation failure or empty mask => frame passes through untouched
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Preset
from .models import ensure_models

# Selfie Multiclass category ids: 0 background, 1 hair, 2 body-skin,
# 3 face-skin, 4 clothes, 5 others.
CLOTHES_CATEGORY = 4

_FEATHER_K = 31


def _feather(mask: np.ndarray, k: int = _FEATHER_K) -> np.ndarray:
    k = k | 1
    return cv2.GaussianBlur(mask, (k, k), 0)


class ClothesSegmenter:
    """MediaPipe ImageSegmenter in VIDEO mode, clothes class only.

    Segmentation runs on a downscaled frame (the model is 256x256 internally
    regardless), and the garment mask is cached and reused between refreshes:
    a garment mask changes slowly, so we only re-segment every `refresh_every`
    calls (monotonic timestamps enforced by MediaPipe are honored by bumping
    the timestamp only on real refreshes).
    """

    _SEG_W = 512  # segmentation input width; height keeps aspect

    def __init__(self, refresh_every: int = 3) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        paths = ensure_models()
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(paths["selfie_multiclass_256x256.tflite"])
            ),
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self._mp = mp
        self._segmenter = vision.ImageSegmenter.create_from_options(opts)
        self._refresh_every = max(1, int(refresh_every))
        self._cached: np.ndarray | None = None
        self._since_refresh = 0
        self._last_ts = -1

    def clothes_mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Feathered uint8 mask (255 = garment) or None if nothing/failed."""
        try:
            h, w = frame_bgr.shape[:2]
            sw = self._SEG_W
            sh = max(64, int(round(h * sw / w)))
            small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
            rgb = small[:, :, ::-1]
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
            )
            result = self._segmenter.segment_for_video(mp_img, max(timestamp_ms, self._last_ts + 1))
            self._last_ts = max(timestamp_ms, self._last_ts + 1)
            cat = result.category_mask.numpy_view().squeeze(-1)
            mask_small = np.where(cat == CLOTHES_CATEGORY, 255, 0).astype(np.uint8)
            # Sliver guard on the small mask (0.5% of segmented area).
            if float(mask_small.mean()) / 255.0 < 0.005:
                full = None
            else:
                # feather at segmentation scale, then upscale (cheaper than full-res blur)
                feathered = cv2.GaussianBlur(mask_small, (9, 9), 0)
                full = cv2.resize(feathered, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None
        return full

    def clothes_mask_cached(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Re-segment every `refresh_every` calls; reuse the mask in between."""
        if self._since_refresh == 0:
            self._cached = self.clothes_mask(frame_bgr, timestamp_ms)
        self._since_refresh = (self._since_refresh + 1) % self._refresh_every
        return self._cached

    def close(self) -> None:
        self._segmenter.close()


def _mask_bbox(mask: np.ndarray, margin: float = 0.05) -> tuple[int, int, int, int] | None:
    """Bounding box of the mask (>=2px per side), or None if too small."""
    ys, xs = np.where(mask > 128)
    if len(xs) < 16:
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w - 1, x1 + mx), min(h - 1, y1 + mx)  # y margin handled above
    y1 = min(h - 1, y1 + my)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _garment_median(roi: np.ndarray, mask_roi: np.ndarray) -> np.ndarray:
    """Median garment color from a strided sample of masked pixels."""
    ys, xs = np.where(mask_roi > 200)
    if len(xs) == 0:
        return np.median(roi.reshape(-1, 3), axis=0)
    step = max(1, len(xs) // 12000)
    pts = roi[ys[::step], xs[::step]].reshape(-1, 3)
    return np.median(pts, axis=0)


def tidy(
    frame_bgr: np.ndarray,
    mask: np.ndarray | None,
    preset: Preset,
) -> np.ndarray:
    """Steam-iron effect. `mask` is the feathered garment mask (full frame)."""
    if mask is None:
        return frame_bgr
    s_clothes = preset.scaled("clothes")
    s_stain = preset.scaled("stain")
    if s_clothes <= 0 and s_stain <= 0:
        return frame_bgr

    box = _mask_bbox(mask)
    if box is None:
        return frame_bgr
    x0, y0, x1, y1 = box
    roi = frame_bgr[y0:y1 + 1, x0:x1 + 1]
    mroi = mask[y0:y1 + 1, x0:x1 + 1]
    out = roi.copy()
    rh, rw = roi.shape[:2]

    # --- crease softening (edge-preserving at half res, applied as a delta layer)
    if s_clothes > 0 and rw >= 4 and rh >= 4:
        hw2, hh2 = rw // 2, rh // 2
        small = cv2.resize(roi, (hw2, hh2), interpolation=cv2.INTER_LINEAR)
        smooth_small = cv2.edgePreservingFilter(small, flags=1, sigma_s=60, sigma_r=0.45)
        a_small = cv2.resize(mroi, (hw2, hh2), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        delta_small = (smooth_small.astype(np.float32) - small.astype(np.float32)) * a_small[..., None] * s_clothes
        delta = cv2.resize(delta_small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        out = np.clip(out.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    # --- stain / color-unevenness pull (low-to-moderate deviations only)
    if s_stain > 0:
        med = _garment_median(roi, mroi).astype(np.float32)
        # deviation computed at half res; the effect is low-frequency by design
        hw2, hh2 = rw // 2, rh // 2
        half = cv2.resize(out, (hw2, hh2), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        m_half = cv2.resize(mroi, (hw2, hh2), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        dev = np.abs(half - med).max(axis=2)
        # ramp: 0 at dev<=12, 1 at dev~55, back toward 0 for extreme deviations
        # (>=150) so crisp prints and stripes survive the steam.
        ramp = np.clip((dev - 12.0) / 43.0, 0, 1) * np.clip((150.0 - dev) / 40.0, 0, 1)
        alpha = m_half * ramp * s_stain * 0.8
        delta_small = (med - half) * alpha[..., None]
        delta = cv2.resize(delta_small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        out = np.clip(out.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    frame_bgr[y0:y1 + 1, x0:x1 + 1] = out
    return frame_bgr
