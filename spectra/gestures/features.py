"""Geometric features derived from hand landmarks.

Finger extension is read from **joint angles** rather than raw tip-versus-joint
coordinate comparisons, so detection keeps working when the hand is tilted or rotated.
The thumb gets its own rule: its interphalangeal angle barely changes between an open
and a closed hand, so abduction (distance from the palm) is used instead.

Every function here is pure and free of OpenCV, NumPy and MediaPipe.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from spectra.detection.landmarks import (
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PALM_TRIANGLE,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
    HandLandmarks,
    LandmarkLike,
)

FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
THUMB, INDEX, MIDDLE, RING, PINKY = range(5)

#: Guard against degenerate (zero-area) detections.
EPSILON = 1e-6

POINTER_ANGLE_THRESHOLD = 120.0
POINTER_Z_TOLERANCE = 0.02


class FingerStates(NamedTuple):
    """Per-finger extension flags, in thumb-to-pinky order."""

    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    @property
    def extended_count(self) -> int:
        return sum(self)

    @property
    def pattern(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


# --------------------------------------------------------------------- geometry
def distance(a: LandmarkLike, b: LandmarkLike) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def joint_angle(a: LandmarkLike, b: LandmarkLike, c: LandmarkLike) -> float:
    """Angle at ``b`` in the triangle ``a-b-c``, in degrees (180 = straight)."""
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    norm = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if norm < EPSILON:
        return 180.0
    cosine = (bax * bcx + bay * bcy) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def palm_size(landmarks: HandLandmarks) -> float:
    """Wrist-to-middle-MCP distance: the scale reference for every ratio."""
    return max(distance(landmarks[WRIST], landmarks[MIDDLE_MCP]), EPSILON)


def finger_extension_angles(landmarks: HandLandmarks) -> tuple[float, ...]:
    """Mean of the two interphalangeal angles per finger, in degrees.

    180 means a fully straight finger; lower values mean more flexion.
    """
    angles = []
    for mcp, pip, dip, tip in FINGER_CHAINS:
        proximal = joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        distal = joint_angle(landmarks[pip], landmarks[dip], landmarks[tip])
        angles.append((proximal + distal) / 2.0)
    return tuple(angles)


def finger_flexion_angles(landmarks: HandLandmarks) -> tuple[float, ...]:
    """Flexion per finger in degrees (0 = straight), the clinical convention."""
    return tuple(180.0 - angle for angle in finger_extension_angles(landmarks))


def thumb_abduction_ratio(landmarks: HandLandmarks) -> float:
    """Thumb-tip distance to the index MCP, normalised by the palm size."""
    return distance(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) / palm_size(landmarks)


def pinch_ratio(landmarks: HandLandmarks) -> float:
    """Thumb-tip to index-tip distance, normalised by the palm size."""
    return distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / palm_size(landmarks)


def opposition_ratios(landmarks: HandLandmarks) -> tuple[float, ...]:
    """Thumb-tip distance to each fingertip (index to pinky), palm-normalised."""
    tips = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    scale = palm_size(landmarks)
    return tuple(distance(landmarks[THUMB_TIP], landmarks[tip]) / scale for tip in tips)


# ------------------------------------------------------- wrist (ESTIMATE ONLY)
def palm_rotation(landmarks: HandLandmarks) -> float:
    """In-plane palm angle in degrees, 0 = fingers pointing straight up.

    **Estimate.** MediaPipe Hands provides no forearm landmarks, so this is the
    orientation of the palm itself, not a true wrist angle. It approximates radial/ulnar
    deviation only while the forearm is held still. See ``docs/metrics.md``.
    """
    wrist = landmarks[WRIST]
    middle = landmarks[MIDDLE_MCP]
    return math.degrees(math.atan2(middle.x - wrist.x, wrist.y - middle.y))


def palm_openness(landmarks: HandLandmarks) -> float:
    """Area of the palm triangle (wrist, index MCP, pinky MCP) over the palm size squared.

    **Estimate.** Foreshortening proxy: the triangle flattens as the palm turns away from
    the camera, so this tracks pronation/supination only in a relative sense.
    """
    a, b, c = (landmarks[i] for i in PALM_TRIANGLE)
    area = abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2.0
    return area / (palm_size(landmarks) ** 2)


# ----------------------------------------------------------------------- pointer
def pointer_position(
    landmarks: HandLandmarks, frame_shape: tuple[int, ...]
) -> tuple[int, int] | None:
    """Index fingertip position in pixels, or ``None`` when the finger is curled."""
    height, width = frame_shape[:2]
    tip = landmarks[INDEX_TIP]
    pip = landmarks[INDEX_PIP]
    extended = (
        joint_angle(landmarks[INDEX_MCP], pip, tip) > POINTER_ANGLE_THRESHOLD
        or tip.z < pip.z - POINTER_Z_TOLERANCE
    )
    if not extended:
        return None
    return (int(tip.x * width), int(tip.y * height))


def extended_count(states: FingerStates) -> int:
    return sum(states)


# ------------------------------------------------------------------ baseline
def finger_states_from_tips(landmarks: HandLandmarks, hand_label: str = "Right") -> FingerStates:
    """Original tip-versus-joint heuristic, kept only as a comparison baseline.

    It is sensitive to hand rotation. Production code uses the angle-based estimator in
    :mod:`spectra.gestures.calibration`.
    """
    tolerance = 0.02
    if hand_label == "Right":
        thumb = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x - tolerance
    else:
        thumb = landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x + tolerance
    return FingerStates(
        thumb=bool(thumb),
        index=landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y - tolerance,
        middle=landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y - tolerance,
        ring=landmarks[RING_TIP].y < landmarks[RING_PIP].y - tolerance,
        pinky=landmarks[PINKY_TIP].y < landmarks[PINKY_PIP].y - tolerance,
    )
