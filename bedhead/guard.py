"""Identity guard: ArcFace embeddings for reference-photo admission + drift cap.

Policy (changed Oct 2026): gallery uploads ARE allowed as reference photos,
but admission is gated: the reference must be similar enough to the live
face (the person appearing on camera) before bedhead will use it. Once
admitted, Tier B rendering will be drift-capped against the same embedding:
if the rendered face drifts too far from the reference identity, fail-safe
to Tier A.

Thresholds were calibrated empirically with buffalo_l ArcFace embeddings:
  same-person pairs (pose/expression/lighting varied): 0.73 - 0.78
  cross-person pairs:                                -0.04 - 0.06
ADMISSION_THRESHOLD = 0.40 sits comfortably between both clusters.

Embedding model: insightface buffalo_l (auto-downloaded to ~/.insightface
on first use, ~200 MiB, on-device inference only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ADMISSION_THRESHOLD = 0.40   # reference must match live face at least this
DRIFT_CAP = 0.35             # Tier B output must stay at least this close to reference
LIVE_SAMPLES = 15            # frames sampled from the live feed for admission


@dataclass
class AdmissionResult:
    admitted: bool
    similarity: float          # best live-sample cosine vs reference
    reason: str


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for (normed) embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class IdentityGuard:
    """ArcFace-backed identity checks (lazily constructs the model)."""

    def __init__(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise RuntimeError(
                "insightface is required for the identity guard; "
                "install with: pip install 'bedhead[guard]'"
            ) from e
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        try:
            import onnxruntime

            avail = onnxruntime.get_available_providers()
            providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
        except ImportError:
            providers = ["CPUExecutionProvider"]
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def embed(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Largest detected face -> normed 512-d embedding, or None."""
        faces = self._app.get(image_bgr)
        if not faces:
            return None
        f = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return f.normed_embedding

    def admit(
        self,
        reference_embedding: np.ndarray,
        live_embeddings: list[np.ndarray],
    ) -> AdmissionResult:
        """Gate: is the reference photo the same person as on camera?

        Uses the BEST live sample: pose/blink variance only lowers individual
        samples, so the max is the robust statistic.
        """
        if reference_embedding is None:
            return AdmissionResult(False, 0.0, "no face found in reference photo")
        if not live_embeddings:
            return AdmissionResult(False, 0.0, "no face found on camera")
        best = max(cosine(reference_embedding, e) for e in live_embeddings)
        if best >= ADMISSION_THRESHOLD:
            return AdmissionResult(
                True, best, f"reference matches live face (similarity {best:.3f})"
            )
        return AdmissionResult(
            False,
            best,
            f"reference photo does not match the face on camera "
            f"(similarity {best:.3f} < {ADMISSION_THRESHOLD}); pick a photo of yourself",
        )

    def drift_ok(self, rendered_embedding: np.ndarray, reference_embedding: np.ndarray) -> bool:
        """Tier B fail-safe: rendered face must stay recognizably the reference."""
        return cosine(rendered_embedding, reference_embedding) >= DRIFT_CAP
