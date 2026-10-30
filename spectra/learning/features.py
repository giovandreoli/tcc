"""Feature vector of a hand pose, invariant to position, size, in-plane rotation and side.

Raw landmarks change whenever the hand moves, so a classifier trained on them would learn
*where* the hand was instead of *what* it was doing. Each pose is therefore normalised:

1. translate so the wrist is the origin;
2. rotate so the wrist→middle-knuckle axis points up;
3. divide by the palm size;
4. mirror left hands, so a gesture learned with one hand is recognised with the other.

The vector holds the 20 normalised (x, y) points plus the five finger-extension angles
(scaled to 0-1), which add robustness when the 2-D outline is ambiguous.
"""

from __future__ import annotations

import math

from spectra.detection.landmarks import MIDDLE_MCP, WRIST, HandLandmarks
from spectra.gestures.features import finger_extension_angles, palm_size

FEATURE_SIZE = 20 * 2 + 5


def pose_features(landmarks: HandLandmarks, hand_label: str = "Right") -> tuple[float, ...]:
    wrist = landmarks[WRIST]
    middle = landmarks[MIDDLE_MCP]
    scale = palm_size(landmarks)
    angle = math.atan2(middle.x - wrist.x, -(middle.y - wrist.y))
    cos, sin = math.cos(angle), math.sin(angle)
    mirror = -1.0 if hand_label == "Left" else 1.0
    values: list[float] = []
    for point in list(landmarks)[1:]:
        dx, dy = (point.x - wrist.x) / scale, (point.y - wrist.y) / scale
        values.extend(((dx * cos + dy * sin) * mirror, -dx * sin + dy * cos))
    values.extend(value / 180.0 for value in finger_extension_angles(landmarks))
    return tuple(values)
