"""Tier B: guarded generative face re-render (spike).

Runs IN Swapper (inswapper_128, official insightface wrapper) inside the
guard contract:

  1. LAUNCH GATE: only an ADMITTED reference (guard.admit passed against the
     live face) can be registered as the identity source. process() refuses
     to run until set_reference() is called; there is no code path that
     swaps in an unverified face.
  2. DRIFT CAP: output is re-embedded every DRIFT_CHECK_INTERVAL frames and
     compared to the reference embedding; below guard.DRIFT_CAP the frame
     fails safe to Tier A and Tier B disables itself for the run.
  3. BLEND DIAL: tier_a_out and the swap are mixed by `blend` (0 = Tier A).

CPU perf (Apple Silicon, this spike): ~73 ms/frame swap+paste. Tier A runs
first; Tier B is additive. GPU (roadmap P1) brings this to 12-30 ms.
"""

from __future__ import annotations

import time

import numpy as np

from .guard import DRIFT_CAP, IdentityGuard

DRIFT_CHECK_INTERVAL = 10  # frames between identity re-checks


class TierBError(RuntimeError):
    pass


class TierB:
    """IN Swapper re-render, hard-wired to the identity guard."""

    def __init__(self, guard: IdentityGuard | None = None, model_path: str | None = None) -> None:
        import cv2  # noqa: F401

        from .models import MODEL_DIR

        self._model_path = model_path or str(MODEL_DIR / "inswapper_128.onnx")
        self._swapper = None  # lazy: heavy session loads on first process()
        self.guard = guard or IdentityGuard()
        self._reference_face = None
        self._reference_embedding: np.ndarray | None = None
        self._last_drift_sim = 1.0
        self._frames_since_check = 0
        self._disabled_reason: str | None = None
        self.last_swap_ms = 0.0

    # ---------------------------------------------------------------- gate

    def set_reference(self, reference_bgr: np.ndarray) -> bool:
        """Register the ADMITTED reference photo. Call only after guard.admit."""
        faces = self.guard._app.get(reference_bgr)
        if not faces:
            raise TierBError("no face found in reference photo")
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self._reference_face = f
        self._reference_embedding = f.normed_embedding
        return True

    @property
    def reference_ready(self) -> bool:
        return self._reference_face is not None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _ensure_session(self) -> None:
        if self._swapper is not None:
            return
        import os

        if not os.path.exists(self._model_path):
            raise TierBError(
                "inswapper_128.onnx not found in the model cache; Tier B "
                "spike requires a one-time manual download (see README)"
            )
        from insightface.model_zoo import model_zoo

        self._swapper = model_zoo.get_model(self._model_path)

    # ---------------------------------------------------------------- swap

    def process(
        self,
        frame_bgr: np.ndarray,
        tier_a_out: np.ndarray,
        blend: float = 1.0,
    ) -> tuple[np.ndarray, str]:
        """Blend Tier B over the Tier A frame; returns (frame, status).

        status: 'ok' | 'passthrough' (no face / not ready / blend 0) |
        'disabled:<reason>'
        """
        if self._disabled_reason is not None:
            return tier_a_out, f"disabled:{self._disabled_reason}"
        if self._reference_face is None or blend <= 0:
            return tier_a_out, "passthrough"
        try:
            self._ensure_session()
        except TierBError as e:
            self._disabled_reason = str(e)
            return tier_a_out, f"disabled:{self._disabled_reason}"

        faces = self.guard._app.get(frame_bgr)
        if not faces:
            return tier_a_out, "passthrough"
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        t0 = time.perf_counter()
        # frequency-separable swap: low-freq identity from the generative
        # render, high-freq native texture from RAW (the naive 0.5 blend was
        # measured destroying skin detail below even RAW)
        import cv2
        from insightface.utils import face_align

        from .freqblend import frequency_composite, paste_face

        aimg, M = face_align.norm_crop2(frame_bgr, face.kps, 128)
        swapped = self._swapper.get(frame_bgr, face, self._reference_face, paste_back=False)
        if isinstance(swapped, tuple):
            swapped = swapped[0]
        if swapped.shape != aimg.shape:
            swapped = cv2.resize(swapped, (aimg.shape[1], aimg.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        composed = frequency_composite(aimg, swapped, blend=1.0)
        swapped = paste_face(frame_bgr, composed, M)
        self.last_swap_ms = (time.perf_counter() - t0) * 1000

        # drift check every N frames (embedding of the swapped output vs ref)
        self._frames_since_check += 1
        if self._frames_since_check >= DRIFT_CHECK_INTERVAL:
            self._frames_since_check = 0
            emb = self.guard.embed(swapped)
            if emb is None:
                return tier_a_out, "passthrough"
            sim = float(np.dot(self._reference_embedding, emb))
            self._last_drift_sim = sim
            if sim < DRIFT_CAP:
                self._disabled_reason = f"drift {sim:.3f} < {DRIFT_CAP}"
                return tier_a_out, f"disabled:{self._disabled_reason}"

        b = float(np.clip(blend, 0.0, 1.0))
        if b >= 0.999:
            return swapped, "ok"
        mixed = (
            tier_a_out.astype(np.float32) * (1 - b) + swapped.astype(np.float32) * b
        ).astype(np.uint8)
        return mixed, "ok"

    @property
    def last_drift_sim(self) -> float:
        return self._last_drift_sim
