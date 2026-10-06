"""Drawing surface, pointer trail and smoothing primitives."""

from spectra.core.canvas import DrawingCanvas
from spectra.core.smoothing import ExponentialSmoother, smooth_point
from spectra.core.trail import Trail

__all__ = ["DrawingCanvas", "ExponentialSmoother", "Trail", "smooth_point"]
