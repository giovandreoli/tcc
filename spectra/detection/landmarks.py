"""Landmark value types and index constants shared by detection and gesture code.

Nothing here imports MediaPipe or OpenCV, so unit tests can build fake hands cheaply.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class LandmarkLike(Protocol):
    """Structural type matching both MediaPipe landmarks and :class:`Landmark`."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Landmark:
    """A normalised landmark, used by tests and by any synthetic hand generator."""

    x: float
    y: float
    z: float = 0.0


#: A full hand is 21 landmarks in MediaPipe's canonical order.
HandLandmarks = Sequence[LandmarkLike]

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

LANDMARK_COUNT = 21

#: (mcp, pip, dip, tip) per finger, in thumb-to-pinky order.
FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)

FINGER_TIPS: tuple[int, ...] = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

#: Landmarks forming the palm plane, used for the (approximate) wrist estimate.
PALM_TRIANGLE = (WRIST, INDEX_MCP, PINKY_MCP)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
