"""Segmentation-driven effects: Studio Light, eye light, background modes.

All compositing is classical OpenCV on the CPU; masks come from
`bedhead.segmenter.Segmenter` (person / face-skin / hair). Everything is
strength-scaled and feathered so nothing can pop.
"""

from __future__ import annotations

import cv2
import numpy as np


def studio_light(frame: np.ndarray, person_mask: np.ndarray, strength: float) -> np.ndarray:
    """Relight the person only: soft luminance lift + slight warmth, feathered.

    Models the "add a ring light" look without touching the background.
    """
    if strength <= 0:
        return frame
    # luminance lift + slight warmth, computed on a copy ...
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan = lab[..., 0].astype(np.float32)
    # gentle highlight-rolling lift, capped
    lifted = l_chan + (245.0 - l_chan) * 0.28 * strength
    lab[..., 0] = np.clip(lifted, 0, 255).astype(np.uint8)
    lit = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    warm = lit * np.array([0.97, 1.0, 1.06], np.float32)
    warm = np.clip(warm, 0, 255)
    # ... then composited strictly inside the (feathered) person mask
    a = person_mask * strength
    return np.clip(
        frame.astype(np.float32) * (1 - a[..., None]) + warm * a[..., None], 0, 255
    ).astype(np.uint8)


def eye_light(
    frame: np.ndarray,
    face_landmarks: np.ndarray | None,
    strength: float,
) -> np.ndarray:
    """Brighten the eye region using landmark rings (independent of segmentation)."""
    if strength <= 0 or face_landmarks is None:
        return frame
    LEFT = (33, 133, 157, 158, 159, 160, 161, 246, 173, 153)
    RIGHT = (263, 362, 398, 384, 385, 386, 387, 466, 380, 374)
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for ring in (LEFT, RIGHT):
        pts = np.round(face_landmarks[list(ring), :2]).astype(np.int32)
        # pad each eye box outward ~40% so lashes/brow-shadow catch light
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = (x1 - x0) * 1.4, (y1 - y0) * 2.2
        p0 = (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)))
        p1 = (int(min(w - 1, cx + bw / 2)), int(min(h - 1, cy + bh / 2)))
        mask[p0[1]:p1[1], p0[0]:p1[0]] = 1.0
    k = 21 | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0) * strength
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[..., 0] = np.clip(
        lab[..., 0].astype(np.float32) + 30.0 * mask, 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def background_mode(
    frame: np.ndarray,
    person_mask: np.ndarray,
    strength: float,
    mode: str,
) -> np.ndarray:
    """Blur or darken the background, person kept crisp (feathered composite)."""
    if strength <= 0 or mode == "off":
        return frame
    h, w = frame.shape[:2]
    if mode == "blur":
        # two-pass: strong blur, then downsample-blur for a camera-like feel
        small = cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(blurred, (0, 0), 2.0)
    elif mode == "dark":
        blurred = (frame.astype(np.float32) * 0.45).astype(np.uint8)
    else:
        return frame
    a = person_mask * strength
    # person in front: keep original where mask high
    return np.clip(
        frame * a[..., None] + blurred * (1 - a[..., None]), 0, 255
    ).astype(np.uint8)
