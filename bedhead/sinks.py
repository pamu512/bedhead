"""Output sinks: virtual camera (pyvirtualcam) and/or preview window.

The pipeline pushes finished frames to any active sink; with none attached it
just runs tracking for stats. Fail-safe: if the virtual camera cannot open
(e.g. OBS not installed on macOS), we degrade to preview-only and say so.
"""

from __future__ import annotations

import numpy as np


class PreviewWindow:
    """cv2 preview window with keyboard shortcuts (built before the Tk UI)."""

    def __init__(self, name: str = "bedhead") -> None:
        import cv2

        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)

    def show(self, frame_bgr: np.ndarray, hud: str | None = None) -> str | None:
        import cv2

        key = None
        if hud:
            frame = frame_bgr.copy()
            for i, line in enumerate(hud.splitlines()):
                cv2.putText(
                    frame, line, (12, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, line, (12, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )
            cv2.imshow(self.name, frame)
        else:
            cv2.imshow(self.name, frame_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            key = chr(k) if 32 <= k < 127 else None
        return key

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(self.name)


class VirtualCamSink:
    """pyvirtualcam sink; on macOS requires OBS with obs-mac-virtualcam plugin."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        import pyvirtualcam

        self._cam = pyvirtualcam.Camera(
            width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False
        )
        print(f"[bedhead] virtual camera live: {self._cam.device} ({width}x{height}@{fps})")
        self.frames_sent = 0

    def send(self, frame_bgr: np.ndarray) -> None:
        self._cam.send(frame_bgr)
        self.frames_sent += 1

    def close(self) -> None:
        self._cam.close()
