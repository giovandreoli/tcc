"""Webcam capture, isolating the Windows-specific DirectShow backend."""

from __future__ import annotations

import cv2
import numpy as np

from spectra.config import is_windows


class Camera:
    """A mirrored webcam stream. Raises :class:`RuntimeError` if it cannot open."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
    ) -> None:
        backend = cv2.CAP_DSHOW if is_windows() else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(index, backend)
        self.mirror = mirror
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        return cv2.flip(frame, 1) if self.mirror else frame

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
