"""Frequency-separable Tier B composite.

The critique measured the 0.5 blend destroying skin texture (cheek HF below
RAW): averaging a 128px generative re-render against the native-res original
cancels high-frequency phase. Fix: composite in frequency bands --

  low frequency (identity structure, color)  <- generative swap
  high frequency (skin texture, pores)       <- RAW frame (native res)

The result keeps the swap's identity/color at the camera's native detail
level. Implementation: guided filter / Gaussian band split on the aligned
face crop, recombination before paste-back.
"""
from __future__ import annotations

import cv2
import numpy as np


def frequency_composite(
    raw: np.ndarray,
    swapped: np.ndarray,
    blend: float = 1.0,
    sigma_low: float = 2.5,
    detail_gain: float = 1.0,
) -> np.ndarray:
    """Recombine: low-freq from `swapped`, high-freq from `raw`.

    Both inputs must be the same shape (aligned face crop, BGR uint8).
    Returns uint8 BGR. `sigma_low` sets the crossover; below it the swap
    rules (structure/color), above it raw rules (texture).
    """
    if raw.shape != swapped.shape:
        raise ValueError(f"shape mismatch: {raw.shape} vs {swapped.shape}")
    raw_f = raw.astype(np.float32)
    swp_f = swapped.astype(np.float32)

    low_raw = cv2.GaussianBlur(raw_f, (0, 0), sigma_low)
    low_swp = cv2.GaussianBlur(swp_f, (0, 0), sigma_low)
    high_raw = (raw_f - low_raw) * detail_gain

    # blend the LOW frequency by `blend` (identity strength), keep raw highs
    low = low_raw * (1 - blend) + low_swp * blend
    out = low + high_raw
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_face(
    frame: np.ndarray,
    face_crop: np.ndarray,
    M: np.ndarray,
    mask_size: int = 128,
) -> np.ndarray:
    """Warp the composed crop back into the frame with a feathered ellipse."""
    h, w = frame.shape[:2]
    warped = cv2.warpAffine(face_crop, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
    m = np.zeros((mask_size, mask_size), np.float32)
    c = mask_size // 2
    cv2.ellipse(m, (c, c), (int(mask_size * 0.47), int(mask_size * 0.47)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (mask_size // 8 | 1, mask_size // 8 | 1), 0)
    mw = cv2.warpAffine(m, M, (w, h))
    out = frame.astype(np.float32) * (1 - mw[..., None]) + warped.astype(np.float32) * mw[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
