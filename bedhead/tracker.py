"""Face tracking: MediaPipe Face Landmarker -> 478-point mesh, iris, blendshapes.

Exposes a per-frame `FaceFrame` with the landmark array plus cached region masks
indices so the retoucher doesn't recompute constants every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .models import ensure_models

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
FOREHEAD_BAND = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                 54, 103, 67, 109)  # placeholder: replaced by dynamic band below

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
    score: float                     # presence score 0..1
    h: int
    w: int
    extras: dict = field(default_factory=dict)

    def poly(self, idx: tuple[int, ...]) -> np.ndarray:
        """Landmark indices -> int32 polygon for cv2.fillPoly."""
        return np.round(self.landmarks[list(idx), :2]).astype(np.int32)

    def box(self, idx: tuple[int, ...], pad: float = 0.0) -> tuple[int, int, int, int]:
        pts = self.landmarks[list(idx), :2]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if pad:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

            def p(v, c, s):  # pad relative to box size
                return int(v - (c - v) * (1 + pad * s))

            x0, y0, x1, y1 = p(x0, cx, 1), p(y0, cy, 1), p(x1, cx, 1), p(y1, cy, 1)
        return int(x0), int(y0), int(x1), int(y1)


class FaceTracker:
    """Wraps MediaPipe FaceLandmarker in VIDEO mode with 1 face."""

    def __init__(self, num_faces: int = 1) -> None:
        paths = ensure_models()
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

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceFrame | None:
        rgb = frame_bgr[:, :, ::-1]  # MediaPipe expects RGB
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        lms = getattr(result, "face_landmarks", None)
        if not lms:
            return None
        h, w = frame_bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h, p.z] for p in lms[0]], dtype=np.float32)
        score = float(np.mean([p.visibility if p.visibility is not None else 1.0 for p in lms[0]]))
        extras: dict = {}
        bs = getattr(result, "face_blendshapes", None)
        if bs:
            categories = {c.category_name: c.score for c in bs[0]}
            extras["jaw_open"] = categories.get("jawOpen", 0.0)
            extras["eye_blink_left"] = categories.get("eyeBlinkLeft", 0.0)
            extras["eye_blink_right"] = categories.get("eyeBlinkRight", 0.0)
        return FaceFrame(landmarks=pts, score=score, h=h, w=w, extras=extras)

    def close(self) -> None:
        self._landmarker.close()
