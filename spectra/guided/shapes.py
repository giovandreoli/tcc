"""Target shapes for guided drawing.

Shapes are defined in **normalised coordinates** (both axes in ``0..1``) so a challenge
scores identically at 640x480 and at 1920x1080. The mode converts to pixels only when
drawing. All geometry here is pure Python: no OpenCV, no NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

Point = tuple[float, float]

#: Resolution of the sampled polyline. Dense enough that the straight segments between
#: samples are shorter than the tightest tolerance corridor.
DEFAULT_SAMPLES = 240


class Difficulty(Enum):
    """Values double as i18n keys under ``guided.difficulty.``."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


#: Half-width of the tolerance corridor, in normalised units.
DIFFICULTY_TOLERANCE: dict[Difficulty, float] = {
    Difficulty.EASY: 0.080,
    Difficulty.MEDIUM: 0.050,
    Difficulty.HARD: 0.030,
}

#: Extra shape complexity per level (zigzag peaks, spiral turns).
DIFFICULTY_COMPLEXITY: dict[Difficulty, int] = {
    Difficulty.EASY: 0,
    Difficulty.MEDIUM: 1,
    Difficulty.HARD: 2,
}


# --------------------------------------------------------------------- geometry
def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Shortest distance from ``point`` to the segment ``start-end``."""
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.hypot(px - ax, py - ay)
    projection = ((px - ax) * dx + (py - ay) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    return math.hypot(px - (ax + projection * dx), py - (ay + projection * dy))


# ------------------------------------------------------------------- the shapes
@dataclass(frozen=True)
class Shape:
    """Base shape; ``key`` doubles as an i18n key under ``guided.shape.``."""

    key: str

    @property
    def closed(self) -> bool:
        """Whether the last sample connects back to the first."""
        return False

    def sample(
        self, difficulty: Difficulty, samples: int = DEFAULT_SAMPLES
    ) -> list[Point]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class LineShape(Shape):
    """A straight stroke. Tilts with difficulty so it stops being purely horizontal."""

    key: str = "line"

    def sample(self, difficulty: Difficulty, samples: int = DEFAULT_SAMPLES) -> list[Point]:
        tilt = 0.12 * DIFFICULTY_COMPLEXITY[difficulty]
        start = (0.12, 0.5 + tilt)
        end = (0.88, 0.5 - tilt)
        return [
            (
                start[0] + (end[0] - start[0]) * i / (samples - 1),
                start[1] + (end[1] - start[1]) * i / (samples - 1),
            )
            for i in range(samples)
        ]


@dataclass(frozen=True)
class CircleShape(Shape):
    """A full circle; the radius shrinks with difficulty, tightening the curve."""

    key: str = "circle"

    @property
    def closed(self) -> bool:
        return True

    def sample(self, difficulty: Difficulty, samples: int = DEFAULT_SAMPLES) -> list[Point]:
        radius = 0.34 - 0.06 * DIFFICULTY_COMPLEXITY[difficulty]
        return [
            (
                0.5 + radius * math.cos(2 * math.pi * i / samples),
                0.5 + radius * math.sin(2 * math.pi * i / samples),
            )
            for i in range(samples)
        ]


@dataclass(frozen=True)
class ZigzagShape(Shape):
    """Alternating peaks; more peaks means more direction reversals per stroke."""

    key: str = "zigzag"

    def peaks(self, difficulty: Difficulty) -> int:
        return 3 + 2 * DIFFICULTY_COMPLEXITY[difficulty]

    def sample(self, difficulty: Difficulty, samples: int = DEFAULT_SAMPLES) -> list[Point]:
        corners = self.peaks(difficulty) + 1
        vertices = [
            (0.12 + 0.76 * i / (corners - 1), 0.28 if i % 2 == 0 else 0.72) for i in range(corners)
        ]
        return _resample_polyline(vertices, samples)


@dataclass(frozen=True)
class SpiralShape(Shape):
    """An Archimedean spiral: the hardest target, with continuously changing curvature."""

    key: str = "spiral"

    def turns(self, difficulty: Difficulty) -> float:
        return 1.5 + 1.0 * DIFFICULTY_COMPLEXITY[difficulty]

    def sample(self, difficulty: Difficulty, samples: int = DEFAULT_SAMPLES) -> list[Point]:
        turns = self.turns(difficulty)
        max_radius = 0.34
        points = []
        for i in range(samples):
            fraction = i / (samples - 1)
            angle = 2 * math.pi * turns * fraction
            radius = max_radius * fraction
            points.append((0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle)))
        return points


def _resample_polyline(vertices: list[Point], samples: int) -> list[Point]:
    """Spread ``samples`` points evenly along a polyline, by arc length."""
    lengths = [math.dist(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]
    total = sum(lengths)
    if total == 0.0:
        return [vertices[0]] * samples
    points: list[Point] = []
    for i in range(samples):
        target = total * i / (samples - 1)
        travelled = 0.0
        for index, length in enumerate(lengths):
            if travelled + length >= target or index == len(lengths) - 1:
                local = (target - travelled) / length if length else 0.0
                local = max(0.0, min(1.0, local))
                ax, ay = vertices[index]
                bx, by = vertices[index + 1]
                points.append((ax + (bx - ax) * local, ay + (by - ay) * local))
                break
            travelled += length
    return points


SHAPES: dict[str, Shape] = {
    shape.key: shape for shape in (LineShape(), ZigzagShape(), CircleShape(), SpiralShape())
}

#: Suggested order of presentation, easiest first.
SHAPE_ORDER: tuple[str, ...] = ("line", "zigzag", "circle", "spiral")


def get_shape(key: str) -> Shape:
    return SHAPES.get(key, SHAPES["line"])


# ------------------------------------------------------------------ target path
@dataclass(frozen=True)
class TargetPath:
    """A sampled shape plus the tolerance corridor the patient must stay inside."""

    shape_key: str
    difficulty: Difficulty
    points: tuple[Point, ...]
    tolerance: float
    closed: bool = False

    @property
    def length(self) -> float:
        segments = list(self.segments())
        return sum(math.dist(a, b) for a, b in segments)

    def segments(self):
        for i in range(len(self.points) - 1):
            yield self.points[i], self.points[i + 1]
        if self.closed and len(self.points) > 1:
            yield self.points[-1], self.points[0]

    def distance_to(self, point: Point) -> float:
        """Shortest distance from ``point`` to the path, in normalised units."""
        return min(point_segment_distance(point, a, b) for a, b in self.segments())

    def nearest_index(self, point: Point) -> int:
        """Index of the closest waypoint, used to track progress along the path."""
        return min(
            range(len(self.points)),
            key=lambda i: math.dist(point, self.points[i]),
        )

    def inside(self, point: Point) -> bool:
        return self.distance_to(point) <= self.tolerance

    def to_pixels(self, width: int, height: int) -> list[tuple[int, int]]:
        return [(int(x * width), int(y * height)) for x, y in self.points]


def build_path(
    shape_key: str,
    difficulty: Difficulty = Difficulty.EASY,
    samples: int = DEFAULT_SAMPLES,
) -> TargetPath:
    """Sample ``shape_key`` at ``difficulty`` into a scorable path."""
    shape = get_shape(shape_key)
    return TargetPath(
        shape_key=shape.key,
        difficulty=difficulty,
        points=tuple(shape.sample(difficulty, samples)),
        tolerance=DIFFICULTY_TOLERANCE[difficulty],
        closed=shape.closed,
    )
