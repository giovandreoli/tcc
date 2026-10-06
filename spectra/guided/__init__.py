"""Target shapes and scoring for the guided-drawing assessment mode."""

from spectra.guided.scoring import TraceResult, TraceScorer
from spectra.guided.shapes import (
    SHAPES,
    CircleShape,
    Difficulty,
    LineShape,
    Shape,
    SpiralShape,
    TargetPath,
    ZigzagShape,
    build_path,
    get_shape,
)

__all__ = [
    "SHAPES",
    "CircleShape",
    "Difficulty",
    "LineShape",
    "Shape",
    "SpiralShape",
    "TargetPath",
    "TraceResult",
    "TraceScorer",
    "ZigzagShape",
    "build_path",
    "get_shape",
]
