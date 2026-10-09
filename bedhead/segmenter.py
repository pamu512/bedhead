"""Person segmentation engine (MediaPipe selfie-multiclass, VIDEO mode).

Runs the segmenter every Nth frame (it costs ~20 ms no matter the input
scale, so it is amortized, not per-frame) and provides temporally smoothed,
feathered person/face-skin/hair masks for:

  - true-skin retouch masking (better than the oval approximation)
  - background blur / replace (Zoom "Portrait" / NVIDIA-style)
  - Studio Light (person-relight without touching the background)
  - eye light (brighten the eye region, "awake" look)

Between segmenter updates the masks are held (they change slowly); the
consumer composites every frame using the latest masks.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import BACKGROUND_CLASS, FACE_SKIN_CLASS, HAIR_CLASS, ensure_models

SEG_INTERVAL = 3          # run the segmenter every Nth processed frame
SEG_INPUT_W, SEG_INPUT_H = 480, 270   # native-ish; cost is scale-invariant
MASK_EMA = 0.6            # temporal smoothing on [0,1] masks (higher = smoother)
FEATHER_K = 31            # gaussian feather on the upsampled mask


class Segmenter:
    """Selfie-multiclass segmentation with temporal smoothing."""

    def __init__(self, interval: int = SEG_INTERVAL) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        paths = ensure_models({"selfie_multiclass_256x256.tflite"})
        base = mp_python.BaseOptions(
            model_asset_path=str(paths["selfie_multiclass_256x256.tflite"])
        )
        opts = vision.ImageSegmenterOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
        )
        self._seg = vision.ImageSegmenter.create_from_options(opts)
        self._mp = None
        self._h = 0
        self._w = 0
        self.interval = max(1, int(interval))
        # float masks at frame resolution, smoothed
        self.person: np.ndarray | None = None      # 1 = person
        self.face_skin: np.ndarray | None = None   # 1 = face skin
        self.hair: np.ndarray | None = None        # 1 = hair
        self._ts = -1
        self._frames_seen = 0

    # ------------------------------------------------------------------ core

    def _update(self, frame_bgr: np.ndarray, ts_ms: int) -> None:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (SEG_INPUT_W, SEG_INPUT_H),
                           interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(small[:, :, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._seg.segment_for_video(mp_img, ts_ms)
        cat = res.category_mask.numpy_view().squeeze()  # (270, 480) uint8

        person_s = (cat != BACKGROUND_CLASS).astype(np.float32)
        face_s = (cat == FACE_SKIN_CLASS).astype(np.float32)
        hair_s = (cat == HAIR_CLASS).astype(np.float32)

        for name, small_mask in (("person", person_s), ("face_skin", face_s),
                                 ("hair", hair_s)):
            up = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            cur = getattr(self, name)
            if (
                cur is None
                or MASK_EMA >= 0.99
                or cur.shape != up.shape  # frame size changed: reset, no EMA
            ):
                setattr(self, name, up)
            else:
                # EMA only where masks exist; hard switch avoids ghosting
                newv = cur * MASK_EMA + up * (1 - MASK_EMA)
                setattr(self, name, newv)

    def tick(
        self,
        frame_bgr: np.ndarray,
        ts_ms: int,
        face_oval_pts: np.ndarray | None = None,
    ) -> None:
        """Call once per processed frame; runs the model every `interval`.

        `face_oval_pts`: optional (N, 2) int polygon of the tracked face oval.
        When provided, the person mask is unioned with the filled oval (with
        the same feather) so the face can never be eaten by background blur
        even if the segmenter under-covers it (observed on some darker skin
        tones where face pixels route to low-confidence classes).
        """
        self._h, self._w = frame_bgr.shape[:2]
        self._oval = face_oval_pts
        self._frames_seen += 1
        if self.person is None or self._frames_seen % self.interval == 0:
            self._ts = max(self._ts + 1, ts_ms)
            self._update(frame_bgr, self._ts)

    def _oval_mask(self, feather: int) -> np.ndarray | None:
        if getattr(self, "_oval", None) is None:
            return None
        m = np.zeros((self._h, self._w), np.float32)
        cv2.fillPoly(m, [np.asarray(self._oval, np.int32)], 1.0)
        k = feather | 1
        return cv2.GaussianBlur(m, (k, k), 0)

    # ------------------------------------------------------------------ sinks

    def person_mask(self, feather: int = FEATHER_K) -> np.ndarray:
        """Feathered person mask [0,1] at frame resolution, oval-unioned."""
        if self.person is None and getattr(self, "_oval", None) is None:
            return np.zeros((self._h, self._w), np.float32)
        k = feather | 1
        base = np.zeros((self._h, self._w), np.float32) if self.person is None \
            else cv2.GaussianBlur(self.person, (k, k), 0)
        ov = self._oval_mask(feather)
        if ov is not None:
            base = np.maximum(base, ov)
        return base

    def face_skin_mask(self, feather: int = FEATHER_K) -> np.ndarray:
        if self.face_skin is None:
            return np.zeros((self._h, self._w), np.float32)
        k = feather | 1
        return cv2.GaussianBlur(self.face_skin, (k, k), 0)

    def hair_mask(self, feather: int = FEATHER_K) -> np.ndarray:
        if self.hair is None:
            return np.zeros((self._h, self._w), np.float32)
        k = feather | 1
        return cv2.GaussianBlur(self.hair, (k, k), 0)

    def close(self) -> None:
        self._seg.close()
