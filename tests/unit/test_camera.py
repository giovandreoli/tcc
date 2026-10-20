"""Webcam backend selection tests that do not require a physical camera."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import spectra.detection.camera as camera_module
from spectra.detection.camera import Camera


class FakeCapture:
    def __init__(self, opened: bool, frame: np.ndarray | None = None) -> None:
        self.opened = opened
        self.frame = frame
        self.released = False
        self.settings = []

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: int) -> None:
        self.settings.append((prop, value))

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.frame is not None, self.frame


def test_windows_camera_falls_back_to_media_foundation(monkeypatch):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    unavailable = FakeCapture(opened=False)
    available = FakeCapture(opened=True, frame=frame)
    attempts = []
    captures = iter((unavailable, available))

    monkeypatch.setattr(camera_module, "is_windows", lambda: True)
    monkeypatch.setattr(
        camera_module.cv2,
        "VideoCapture",
        lambda index, backend: attempts.append((index, backend)) or next(captures),
    )

    camera = Camera(index=2, width=640, height=480, fps=25, mirror=False)

    assert attempts == [(2, cv2.CAP_DSHOW), (2, cv2.CAP_MSMF)]
    assert unavailable.released is True
    assert camera.read() is frame
    assert available.settings == [
        (cv2.CAP_PROP_FRAME_WIDTH, 640),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480),
        (cv2.CAP_PROP_FPS, 25),
    ]
    camera.release()
    assert available.released is True


def test_non_windows_uses_automatic_backend(monkeypatch):
    available = FakeCapture(opened=True)
    attempts = []

    monkeypatch.setattr(camera_module, "is_windows", lambda: False)
    monkeypatch.setattr(
        camera_module.cv2,
        "VideoCapture",
        lambda index, backend: attempts.append((index, backend)) or available,
    )

    camera = Camera(index=1)

    assert attempts == [(1, cv2.CAP_ANY)]
    camera.release()


def test_camera_error_lists_attempted_backends_and_releases_captures(monkeypatch):
    unavailable = [FakeCapture(opened=False) for _ in range(3)]
    captures = iter(unavailable)
    monkeypatch.setattr(camera_module, "is_windows", lambda: True)
    monkeypatch.setattr(
        camera_module.cv2,
        "VideoCapture",
        lambda index, backend: next(captures),
    )

    with pytest.raises(RuntimeError, match="DirectShow.*Media Foundation.*automatic"):
        Camera()

    assert all(capture.released for capture in unavailable)
