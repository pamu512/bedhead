"""Best-in-class quality primitives: one-euro filtering, frequency separation.

Everything here is pure NumPy/OpenCV, O(N) or box-filter based, designed for
<5 ms total at 720p ROI on Apple Silicon CPU.

References:
  - One-Euro filter: Casiez & Roussel, CHI 2012
    https://dl.acm.org/doi/10.1145/2207676.2208639
  - Guided filter: He et al., ECCV 2010 / PAMI 2012 (opencv-contrib guidedFilter)
    https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf
  - Frequency separation: standard retouching practice (low-freq base layer +
    high-freq detail layer, attenuate only mid-frequency blemishes).
"""
from __future__ import annotations

import cv2
import numpy as np


class OneEuro:
    """Vectorized one-euro filter for N landmark points (x, y per point).

    Kills low-speed jitter with near-zero lag on fast motion. Stateful.
    """

    def __init__(
        self,
        n_points: int,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._prev: np.ndarray | None = None          # (n, 2)
        self._prev_filt: np.ndarray | None = None     # (n, 2)
        self._d_prev: np.ndarray | None = None        # (n, 2) smoothed speed

    def __call__(self, pts_xy: np.ndarray) -> np.ndarray:
        """pts_xy: (n, 2) float32 -> filtered (n, 2) float32."""
        if self._prev is None:
            self._prev = pts_xy.copy()
            self._prev_filt = pts_xy.copy()
            self._d_prev = np.zeros_like(pts_xy)
            return pts_xy.copy()
        d_prev = self._d_prev
        prev_filt = self._prev_filt
        assert d_prev is not None and prev_filt is not None
        dx = (pts_xy - self._prev) * self.freq                     # speed per point
        tau_d = 1.0 / (2.0 * np.pi * self.d_cutoff)
        a_d = 1.0 / (1.0 + tau_d * self.freq)                      # scalar
        d_hat = a_d * dx + (1.0 - a_d) * d_prev                    # smoothed speed
        cutoff = (self.min_cutoff + self.beta * np.abs(d_hat)).astype(np.float32)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau * self.freq)                          # (n, 1)
        out = prev_filt + a * (pts_xy - prev_filt)
        self._prev = pts_xy.copy()
        self._prev_filt = out.copy()
        self._d_prev = d_hat
        return out


class Schmitt:
    """Two-threshold hysteresis for a scalar gate (stops flicker at boundaries)."""

    def __init__(self, hi: float, lo: float, initial: bool = False) -> None:
        self.hi = hi
        self.lo = lo
        self.state = initial

    def __call__(self, v: float) -> bool:
        if v >= self.hi:
            self.state = True
        elif v < self.lo:
            self.state = False
        return self.state


def frequency_separation(
    roi_bgr: np.ndarray,
    radius: int = 8,
    eps: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Guided-filter low-frequency base + high-frequency detail, on LAB L.

    Returns (base_l, detail_l) as float32 in [0, 255+] range; base is the
    edge-aware smooth layer, detail = L - base carries pores/edges/blemishes.
    """
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    li = lab[..., 0].astype(np.float32)
    base = cv2.ximgproc.guidedFilter(
        guide=lab[..., 0], src=li, radius=radius, eps=eps
    )
    detail = li - base
    return base, detail


def unsharp_detail(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Sharpen eyes/lips/brows: add scaled high-frequency residual inside mask."""
    if amount <= 0 or not mask.any():
        return roi_bgr
    blur = cv2.GaussianBlur(roi_bgr, (0, 0), 2.0)
    residual = roi_bgr.astype(np.float32) - blur.astype(np.float32)
    a = (mask.astype(np.float32) / 255.0)[..., None] * amount
    return np.clip(
        roi_bgr.astype(np.float32) + residual * a, 0, 255
    ).astype(np.uint8)
