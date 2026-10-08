"""Retoucher behavior: passthrough guarantees, real tracking, ROI clipping."""

import cv2
import numpy as np
import pytest

from bedhead.config import Preset
from bedhead.retoucher import _retouch_roi, _roi, apply
from bedhead.tracker import FACE_OVAL, FaceFrame, FaceTracker

H, W = 480, 640


def synthetic_face_frame(h: int = H, w: int = W) -> np.ndarray:
    """Draw a synthetic face the real tracker can lock onto (478 points).

    Proven against FaceLandmarker: skin oval + sclera/pupils + brows + mouth.
    If detection ever breaks, tune the geometry here until it detects again.
    """
    frame = np.full((h, w, 3), (70, 110, 150), np.uint8)  # bluish-gray backdrop
    skin = (140, 175, 205)  # BGR skin tone
    cv2.ellipse(frame, (w // 2, int(h * 0.55)), (120, 160), 0, 0, 360, skin, -1)
    for cx in (w // 2 - 55, w // 2 + 55):  # sclera + pupil
        cv2.ellipse(frame, (cx, int(h * 0.48)), (24, 14), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(frame, (cx, int(h * 0.48)), 7, (40, 40, 40), -1)
    for cx in (w // 2 - 55, w // 2 + 55):  # brows
        cv2.ellipse(frame, (cx, int(h * 0.42)), (28, 8), 0, 0, 360, (60, 75, 110), -1)
    cv2.ellipse(frame, (w // 2, int(h * 0.70)), (45, 22), 0, 0, 360, (60, 60, 130), -1)  # mouth
    cv2.circle(frame, (w // 2, int(h * 0.585)), 8, (125, 155, 185), -1)  # nose
    return frame


def gradient_frame(h: int = H, w: int = W) -> np.ndarray:
    """Deterministic non-trivial pixel content for byte-exact passthrough checks."""
    return np.arange(h * w * 3, dtype=np.uint64).reshape(h, w, 3).astype(np.uint8)


def oval_face(center: tuple[float, float], axes: tuple[float, float]) -> FaceFrame:
    """FaceFrame whose FACE_OVAL landmarks sit on an ellipse at ``center``.

    All other landmarks default to the center point; only ROI/mask geometry
    cares about the oval placement.
    """
    cx, cy = center
    landmarks = np.full((478, 3), (cx, cy, 0.0), np.float32)
    n = len(FACE_OVAL)
    for k, idx in enumerate(FACE_OVAL):
        angle = 2 * np.pi * k / n
        landmarks[idx, 0] = cx + axes[0] * np.cos(angle)
        landmarks[idx, 1] = cy + axes[1] * np.sin(angle)
    return FaceFrame(landmarks=landmarks, score=1.0, h=H, w=W)


def centered_face() -> FaceFrame:
    """A well-formed in-frame FaceFrame without running the tracker."""
    return oval_face((W / 2, H / 2), (120, 160))


# ---------------------------------------------------------------- passthrough


class TestPassthrough:
    def test_no_face_returns_input_bytes_unchanged(self):
        frame = gradient_frame()
        out = apply(frame, None, Preset())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert out.tobytes() == frame.tobytes()

    @pytest.mark.parametrize("intensity", [0.0, -0.4])
    def test_non_positive_intensity_returns_input_bytes(self, intensity):
        frame = gradient_frame()
        out = apply(frame, centered_face(), Preset(intensity=intensity))
        assert out.tobytes() == frame.tobytes()

    def test_show_original_returns_input_bytes(self):
        frame = gradient_frame()
        out = apply(frame, centered_face(), Preset(show_original=True))
        assert out.tobytes() == frame.tobytes()

    def test_nan_intensity_returns_input_bytes(self):
        frame = gradient_frame()
        out = apply(frame, centered_face(), Preset(intensity=float("nan")))
        assert out.tobytes() == frame.tobytes()

    def test_negative_infinity_intensity_returns_input_bytes(self):
        frame = gradient_frame()
        out = apply(frame, centered_face(), Preset(intensity=float("-inf")))
        assert out.tobytes() == frame.tobytes()


# ------------------------------------------------------------ real detection


@pytest.fixture(scope="module")
def tracker(models):
    t = FaceTracker()
    yield t
    t.close()


@pytest.fixture(scope="module")
def detected_face(tracker):
    frame = synthetic_face_frame()
    face = tracker.detect(frame, 0)
    assert face is not None, (
        "synthetic face drawing not detected - adjust synthetic_face_frame()"
    )
    return frame, face


def test_real_tracker_yields_478_point_mesh(detected_face):
    _, face = detected_face
    assert isinstance(face, FaceFrame)
    assert face.landmarks.shape == (478, 3)


def test_apply_default_preset_on_detected_face(detected_face):
    frame, face = detected_face
    out = apply(frame, face, Preset())
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- ROI clipping


class TestRoiClipping:
    def test_roi_invariants_face_past_left_top(self):
        face = oval_face((60, 50), (110, 150))  # oval spans x -50..170, y -100..200
        x0, y0, x1, y1 = _roi(face)
        assert 0 <= x0 <= x1 < face.w
        assert 0 <= y0 <= y1 < face.h

    def test_roi_invariants_face_past_right_bottom(self):
        face = oval_face((580, 430), (110, 150))  # oval spans x 470..690, y 280..580
        x0, y0, x1, y1 = _roi(face)
        assert 0 <= x0 <= x1 < face.w
        assert 0 <= y0 <= y1 < face.h

    def test_roi_invariants_face_fully_outside_frame(self):
        face = oval_face((-200.0, -200.0), (110, 150))  # entire oval off-screen
        x0, y0, x1, y1 = _roi(face)
        assert 0 <= x0 <= x1 < face.w
        assert 0 <= y0 <= y1 < face.h

    @pytest.mark.parametrize(
        "center",
        [(60, 50), (580, 430), (-200.0, -200.0), (900.0, 700.0)],
    )
    def test_apply_off_edge_face_keeps_frame_valid(self, center):
        frame = gradient_frame()
        face = oval_face(center, (110, 150))
        out = apply(frame, face, Preset())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert np.isfinite(out).all()

    def test_apply_collapsed_oval_keeps_frame_valid(self):
        frame = gradient_frame()
        face = oval_face((10.0, 10.0), (0.0, 0.0))
        out = apply(frame, face, Preset())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert np.isfinite(out).all()

    def test_apply_one_pixel_frame_keeps_frame_valid(self):
        frame = np.zeros((1, 1, 3), np.uint8)
        landmarks = np.zeros((478, 3), np.float32)
        face = FaceFrame(landmarks=landmarks, score=1.0, h=1, w=1)
        out = apply(frame, face, Preset())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert np.isfinite(out).all()

    def test_retouch_roi_under_two_pixels_skips_resize(self):
        roi = np.full((1, 8, 3), 90, np.uint8)
        landmarks = np.full((478, 3), (1.0, 0.0, 0.0), np.float32)
        face = FaceFrame(landmarks=landmarks, score=1.0, h=1, w=8)
        out = _retouch_roi(roi, face, Preset(intensity=1.0))
        assert out.tobytes() == roi.tobytes()

    def test_retouch_roi_two_pixel_box_stays_finite(self):
        roi = np.full((2, 2, 3), 128, np.uint8)
        landmarks = np.full((478, 3), (1.0, 1.0, 0.0), np.float32)
        face = FaceFrame(landmarks=landmarks, score=1.0, h=2, w=2)
        out = _retouch_roi(roi, face, Preset(intensity=1.0))
        assert out.shape == roi.shape
        assert out.dtype == np.uint8
        assert np.isfinite(out).all()

    def test_apply_positive_infinity_intensity_stays_finite(self):
        frame = gradient_frame()
        out = apply(frame, centered_face(), Preset(intensity=float("inf")))
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert np.isfinite(out).all()
