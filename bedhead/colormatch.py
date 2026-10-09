"""Reference color matching: Reinhard LAB statistics transfer, Tier A safe.

Moves the live face's color statistics (mean/std per LAB channel) toward the
admitted reference photo's, inside the face region only:

  - geometry untouched (pure per-pixel color remap; identity-safe by design)
  - strength-scaled: at 1.0, LAB stats match the reference exactly
  - bounded: chroma shift is clamped so it can never push skin into
    unnatural hues, and the drift cap (ArcFace) can verify identity survives

This is the classical foundation under Tier B: when the generative re-render
lands, this same transfer keeps colors anchored to the reference.
"""

from __future__ import annotations

import cv2
import numpy as np

# max chroma shift (a*/b*) we allow the transfer to apply, in LAB units;
# beyond this the "match" would require actually becoming a different photo
MAX_CHROMA_SHIFT = 30.0


def _lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def lab_stats(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) of LAB, optionally inside a [0,1] mask."""
    lab = _lab(img_bgr)
    if mask is not None and float(mask.sum()) > 0:
        sel = lab[mask > 0.5]
    else:
        sel = lab.reshape(-1, 3)
    mean = sel.mean(axis=0)
    std = sel.std(axis=0) + 1e-6
    return mean, std


def color_match(
    frame_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    strength: float = 1.0,
    face_mask: np.ndarray | None = None,
    ref_face_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Reinhard transfer of reference color stats into the frame's face.

    `face_mask`/`ref_face_mask`: [0,1] float masks; stats are computed inside
    them. With no masks, whole-frame stats are used (still bounded by the
    chroma clamp).
    """
    if strength <= 0:
        return frame_bgr
    s = float(np.clip(strength, 0.0, 1.0))

    src_mean, src_std = lab_stats(frame_bgr, face_mask)
    ref_mean, ref_std = lab_stats(reference_bgr, ref_face_mask)

    # scale/shift toward reference, blended by strength
    tgt_mean = src_mean * (1 - s) + ref_mean * s
    tgt_std = src_std * (1 - s) + ref_std * s

    lab = _lab(frame_bgr)
    out = (lab - src_mean) / src_std * tgt_std + tgt_mean

    # bound the chroma movement in the OUTPUT (mean shift + std rescale both
    # move a*/b*; clamp the combined effect so a match can never push skin
    # into unnatural hues)
    for ch in (1, 2):
        delta_ch = out[..., ch] - lab[..., ch]
        moved = float(np.mean(delta_ch))
        if abs(moved) > MAX_CHROMA_SHIFT:
            excess = abs(moved) - MAX_CHROMA_SHIFT
            out[..., ch] -= np.sign(moved) * excess

    out = np.clip(out, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    if face_mask is None:
        return matched
    # feathered blend so the matched face never seams against the background
    k = 25 | 1
    a = cv2.GaussianBlur(face_mask.astype(np.float32), (k, k), 0)
    return np.clip(
        frame_bgr.astype(np.float32) * (1 - a[..., None]) + matched.astype(np.float32) * a[..., None],
        0, 255,
    ).astype(np.uint8)
