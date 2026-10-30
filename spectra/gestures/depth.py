"""Depth (Z) estimation from a single 2-D camera, using the palm polygon.

A webcam gives no depth, but the hand's projected size does: under a pinhole camera a hand
twice as far looks half as long and a quarter as large. The **palm polygon** (wrist, thumb
base and the four finger knuckles) is used because it is nearly rigid: bending the fingers
does not change it, unlike the outline of the whole hand.

Area alone is ambiguous. Tilting the hand also shrinks the projected area (by the cosine of
the tilt), which a naive estimator reads as "the hand moved away". Two complementary
metrics separate the cases, as proposed in the project:

* **Aspect ratio** (width over height of the polygon, measured along the hand's own axis).
  Moving in Z scales width and height equally, so the ratio is unchanged; tilting shortens
  only one of them, so the ratio changes. Under a single-axis tilt the ratio changes by
  exactly the cosine of the tilt, which also lets the area be corrected.
* **Wrist to middle-finger knuckle distance.** Moving in Z scales it like any length (by
  ``1/distance``), while rotating the palm about its long axis (pronation/supination) leaves
  it almost unchanged. Comparing it with the square root of the area tells the two apart.

Every estimate is **relative** to a reference captured at a comfortable position
(``scale = 1`` there; ``scale > 1`` means closer). An absolute distance follows only when the
reference distance is known, and is an approximation.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum

from spectra.detection.landmarks import (
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    RING_MCP,
    THUMB_CMC,
    WRIST,
    HandLandmarks,
)

#: Palm polygon, in order around its outline.
PALM_POLYGON = (WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
EPSILON = 1e-9
#: Relative aspect change below which a size change is treated as pure translation.
ROTATION_TOLERANCE = 0.12
#: Relative mismatch between area-scale and length-scale tolerated as translation.
LENGTH_TOLERANCE = 0.10
REFERENCE_FRAMES = 15


class DepthMethod(Enum):
    """Estimators compared in the project; values double as i18n keys."""

    AREA = "area"
    AREA_ASPECT = "area_aspect"
    AREA_LENGTH = "area_length"


class Motion(Enum):
    """What changed since the reference; values double as i18n keys."""

    STEADY = "steady"
    TRANSLATION = "translation"
    ROTATION = "rotation"


@dataclass(frozen=True)
class PalmGeometry:
    """Scale-dependent and scale-free measurements of the palm polygon."""

    area: float
    length: float
    width: float
    height: float

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, EPSILON)


def shoelace_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def palm_geometry(landmarks: HandLandmarks, image_aspect: float = 1.0) -> PalmGeometry:
    """Measure the palm polygon in a frame of the hand's own axis.

    ``image_aspect`` is width/height of the image, so normalised x and y are comparable.
    Width and height are the extents of the polygon across and along the wrist→middle-knuckle
    axis, which makes the aspect ratio independent of how the hand is turned in the image.
    """
    points = [(landmarks[i].x * image_aspect, landmarks[i].y) for i in PALM_POLYGON]
    wrist, middle = points[0], points[3]
    axis_x, axis_y = middle[0] - wrist[0], middle[1] - wrist[1]
    length = math.hypot(axis_x, axis_y)
    if length < EPSILON:
        return PalmGeometry(0.0, 0.0, 0.0, 0.0)
    ux, uy = axis_x / length, axis_y / length
    along = [(x - wrist[0]) * ux + (y - wrist[1]) * uy for x, y in points]
    across = [-(x - wrist[0]) * uy + (y - wrist[1]) * ux for x, y in points]
    return PalmGeometry(
        area=shoelace_area(points),
        length=length,
        width=max(across) - min(across),
        height=max(along) - min(along),
    )


@dataclass(frozen=True)
class DepthReading:
    """One depth estimate relative to the reference pose."""

    scale: float
    motion: Motion
    method: DepthMethod
    #: Estimated tilt (degrees) explaining the foreshortening, when it is a rotation.
    tilt_degrees: float = 0.0

    @property
    def relative_distance(self) -> float:
        """Distance divided by the reference distance (1 = same, 2 = twice as far)."""
        return 1.0 / max(self.scale, EPSILON)

    def distance_cm(self, reference_cm: float) -> float:
        return reference_cm * self.relative_distance


def _classify(scale: float, rotated: bool) -> Motion:
    if rotated:
        return Motion.ROTATION
    return Motion.STEADY if abs(math.log(max(scale, EPSILON))) < 0.05 else Motion.TRANSLATION


def estimate(geometry: PalmGeometry, reference: PalmGeometry, method: DepthMethod) -> DepthReading:
    """Depth estimate of ``geometry`` relative to ``reference`` with one of the methods."""
    area_scale = math.sqrt(geometry.area / max(reference.area, EPSILON))
    if method is DepthMethod.AREA:
        return DepthReading(area_scale, _classify(area_scale, False), method)

    if method is DepthMethod.AREA_ASPECT:
        change = geometry.aspect / max(reference.aspect, EPSILON)
        foreshortening = min(change, 1.0 / max(change, EPSILON))
        rotated = 1.0 - foreshortening > ROTATION_TOLERANCE
        if not rotated:
            return DepthReading(area_scale, _classify(area_scale, False), method)
        # Under a tilt the area shrinks by cos(tilt) and the aspect changes by the same
        # factor, so dividing it out recovers the scale due to distance alone.
        scale = area_scale / math.sqrt(max(foreshortening, EPSILON))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, foreshortening))))
        return DepthReading(scale, Motion.ROTATION, method, tilt)

    length_scale = geometry.length / max(reference.length, EPSILON)
    mismatch = area_scale / max(length_scale, EPSILON)
    if abs(math.log(max(mismatch, EPSILON))) <= LENGTH_TOLERANCE:
        scale = math.sqrt(area_scale * length_scale)
        return DepthReading(scale, _classify(scale, False), method)
    # The wrist-middle length is the more rotation-stable of the two measures.
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, mismatch**2)))) if mismatch < 1 else 0.0
    return DepthReading(length_scale, Motion.ROTATION, method, tilt)


class DepthEstimator:
    """Stateful estimator: captures a reference, then reports smoothed depth readings."""

    def __init__(
        self,
        method: DepthMethod = DepthMethod.AREA_ASPECT,
        image_aspect: float = 1.0,
        smoothing: float = 0.3,
        reference_frames: int = REFERENCE_FRAMES,
    ) -> None:
        self.method = method
        self.image_aspect = image_aspect
        self.smoothing = smoothing
        self.reference_frames = max(1, reference_frames)
        self.reference: PalmGeometry | None = None
        self._pending: list[PalmGeometry] = []
        self._scale: float | None = None
        self.reading: DepthReading | None = None

    @property
    def calibrated(self) -> bool:
        return self.reference is not None

    def recalibrate(self) -> None:
        self.reference = None
        self._pending.clear()
        self._scale = None
        self.reading = None

    def set_reference(self, geometry: PalmGeometry) -> None:
        self.reference = geometry
        self._pending.clear()
        self._scale = None

    def update(self, landmarks: HandLandmarks | None) -> DepthReading | None:
        if landmarks is None:
            self.reading = None
            return None
        geometry = palm_geometry(landmarks, self.image_aspect)
        if geometry.area <= EPSILON:
            return self.reading
        if self.reference is None:
            self._pending.append(geometry)
            if len(self._pending) < self.reference_frames:
                return None
            self.reference = PalmGeometry(
                area=statistics.median(g.area for g in self._pending),
                length=statistics.median(g.length for g in self._pending),
                width=statistics.median(g.width for g in self._pending),
                height=statistics.median(g.height for g in self._pending),
            )
            self._pending.clear()
        raw = estimate(geometry, self.reference, self.method)
        alpha = self.smoothing
        self._scale = (
            raw.scale if self._scale is None else (1 - alpha) * self._scale + alpha * raw.scale
        )
        self.reading = DepthReading(self._scale, raw.motion, raw.method, raw.tilt_degrees)
        return self.reading
