"""Face tracking: MediaPipe Face Landmarker -> 478-point mesh, iris, blendshapes.

Exposes a per-frame `FaceFrame` with the landmark array plus cached region masks
indices so the retoucher doesn't recompute constants every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .models import ensure_models
from .quality import OneEuro, Schmitt

# ---------------------------------------------------------------- region indices
# MediaPipe FaceMesh canonical topology (see editor.pixees.artifacts / mediapipe docs).
# Kept as plain tuples so they can be dumped into cv2.fillPoly in one call.

# Approximate face-oval inner offset = skin area minus features. We build the
# skin polygon from the face oval, then subtract eyes/lips/brows via mask ops.
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
)
LEFT_EYE_RING = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE_RING = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
LEFT_BROW = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
RIGHT_BROW = (300, 293, 334, 296, 313, 276, 283, 282, 295, 285)
OUTER_LIPS = (61, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
INNER_LIPS = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14)
# Landmarks whose neighborhood is treated as "skin" for tone stats.
CHEEK_SAMPLES = (
    50, 101, 119, 36, 205, 58, 355, 465, 340, 326, 316,  448,  465, 330, 349,
    280, 351, 425, 266, 2,
)
FOREHEAD_SAMPLES = (10, 151, 9, 8, 107, 336, 66, 296, 109, 338, 246, 410)


@dataclass
class FaceFrame:
    """Everything the retoucher needs for one video frame."""

    landmarks: np.ndarray            # (478, 3) image-pixel coords
    h: int
    w: int
    extras: dict = field(default_factory=dict)

    def poly(self, idx: tuple[int, ...]) -> np.ndarray:
        """Landmark indices -> int32 polygon for cv2.fillPoly."""
        return np.round(self.landmarks[list(idx), :2]).astype(np.int32)


class FaceTracker:
    """Wraps MediaPipe FaceLandmarker in VIDEO mode with 1 face."""

    def __init__(self, num_faces: int = 1) -> None:
        paths = ensure_models({"face_landmarker.task"})
        base = mp_python.BaseOptions(model_asset_path=str(paths["face_landmarker.task"]))
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            # Blendshapes let us gate teeth whitening on jaw open; kept lightweight.
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        # temporal quality: one-euro on the 478 (x, y) landmarks, hysteresis on
        # the jaw-open gate so the teeth effect never flickers at half-open.
        self._euro = OneEuro(478, freq=30.0, min_cutoff=1.2, beta=0.02)
        self._jaw_gate = Schmitt(hi=0.30, lo=0.18)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceFrame | None:
        rgb = frame_bgr[:, :, ::-1]  # MediaPipe expects RGB
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        lms = getattr(result, "face_landmarks", None)
        if not lms:
            return None
        h, w = frame_bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h, p.z] for p in lms[0]], dtype=np.float32)
        # temporal smoothing: one-euro on x, y only (z is unused downstream)
        smoothed = self._euro(pts[:, :2].copy())
        pts[:, :2] = smoothed
        extras: dict = {}
        bs = getattr(result, "face_blendshapes", None)
        if bs:
            categories = {c.category_name: c.score for c in bs[0]}
            extras["jaw_open"] = categories.get("jawOpen", 0.0)
            extras["eye_blink_left"] = categories.get("eyeBlinkLeft", 0.0)
            extras["eye_blink_right"] = categories.get("eyeBlinkRight", 0.0)
            extras["jaw_open_gated"] = 1.0 if self._jaw_gate(
                categories.get("jawOpen", 0.0)) else 0.0
        return FaceFrame(landmarks=pts, h=h, w=w, extras=extras)

    def close(self) -> None:
        self._landmarker.close()
