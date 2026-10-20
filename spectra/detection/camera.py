"""Webcam capture with fallback across Windows capture backends."""

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
        self.mirror = mirror
        self._capture = self._open_capture(index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)

    @staticmethod
    def _open_capture(index: int) -> cv2.VideoCapture:
        """Try Windows capture backends in order before reporting an open failure."""
        backends = (
            (
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "automatic"),
            )
            if is_windows()
            else ((cv2.CAP_ANY, "automatic"),)
        )
        errors = []
        for backend, name in backends:
            try:
                capture = cv2.VideoCapture(index, backend)
            except cv2.error as exc:
                errors.append(f"{name}: {exc}")
                continue
            if capture.isOpened():
                return capture
            capture.release()
            errors.append(f"{name}: device did not open")

        details = "; ".join(errors)
        raise RuntimeError(f"could not open camera {index}; tried {details}")

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
