"""Synthetic 3-D hand and pinhole camera, for evaluations with a known ground truth.

A real webcam cannot tell how far the hand actually was, so the depth and smoothing methods
are first measured on a simulated hand whose true pose is known exactly. The hand is a flat
model of an adult palm and fingers (centimetres), posed by a rotation and a distance, and
projected with a pinhole camera of a typical laptop webcam (about 60° horizontal field of
view). Gaussian noise of about 2 px reproduces MediaPipe's landmark jitter.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from spectra.detection.landmarks import Landmark

#: Adult right hand, palm facing the camera, fingers extended, wrist at the origin (cm).
#: x to the image right, y to the image bottom, z towards the camera = 0.
HAND_MODEL_CM: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (-2.4, -2.0),
    (-4.0, -3.6),
    (-5.2, -5.2),
    (-6.0, -6.6),
    (-2.5, -8.6),
    (-2.7, -12.2),
    (-2.8, -14.4),
    (-2.9, -16.4),
    (-0.4, -9.0),
    (-0.4, -13.0),
    (-0.4, -15.4),
    (-0.4, -17.5),
    (1.6, -8.6),
    (1.8, -12.2),
    (1.9, -14.5),
    (2.0, -16.4),
    (3.4, -7.6),
    (3.8, -10.4),
    (4.0, -12.2),
    (4.1, -13.8),
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOCAL_PX = 555.0  # about 60° horizontal field of view at 640 px
PALM_CENTER_CM = (-0.2, -7.0)


@dataclass(frozen=True)
class HandPose:
    """Distance from the camera plus rotations about the palm's own axes (degrees).

    ``roll`` turns the palm about its long axis (pronation/supination), ``pitch`` tips the
    fingers towards or away from the camera, ``yaw`` turns the hand within the image plane.
    """

    distance_cm: float = 50.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    offset_cm: tuple[float, float] = (0.0, 0.0)


def _rotate(point: tuple[float, float, float], pose: HandPose) -> tuple[float, float, float]:
    x, y, z = point
    r, p, w = (math.radians(value) for value in (pose.roll, pose.pitch, pose.yaw))
    # roll about the palm's long axis (y)
    x, z = x * math.cos(r) + z * math.sin(r), -x * math.sin(r) + z * math.cos(r)
    # pitch about the across-palm axis (x)
    y, z = y * math.cos(p) - z * math.sin(p), y * math.sin(p) + z * math.cos(p)
    # yaw in the image plane (z)
    x, y = x * math.cos(w) - y * math.sin(w), x * math.sin(w) + y * math.cos(w)
    return x, y, z


def project(
    pose: HandPose, noise_px: float = 0.0, rng: random.Random | None = None
) -> list[Landmark]:
    """Normalised landmarks (x, y in 0..1 of the image) of the hand in ``pose``."""
    rng = rng or random.Random(0)
    cx, cy = PALM_CENTER_CM
    landmarks = []
    for x, y in HAND_MODEL_CM:
        rx, ry, rz = _rotate((x - cx, y - cy, 0.0), pose)
        depth = pose.distance_cm - rz
        u = IMAGE_WIDTH / 2 + FOCAL_PX * (rx + pose.offset_cm[0]) / depth
        v = IMAGE_HEIGHT / 2 + FOCAL_PX * (ry + pose.offset_cm[1]) / depth
        if noise_px:
            u += rng.gauss(0.0, noise_px)
            v += rng.gauss(0.0, noise_px)
        landmarks.append(Landmark(u / IMAGE_WIDTH, v / IMAGE_HEIGHT, -rz / 100.0))
    return landmarks


IMAGE_ASPECT = IMAGE_WIDTH / IMAGE_HEIGHT
