"""Reproducible evaluation of the pointer filters and the depth estimators.

Run ``python -m spectra.evaluation`` to print the tables used in the thesis. Every number
comes from synthetic data with a fixed random seed, so it can be regenerated exactly; the
user tests then confirm (or not) the same ranking with real hands.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from spectra.core.kalman import make_point_filter
from spectra.evaluation.synthetic import IMAGE_ASPECT, HandPose, project
from spectra.gestures.depth import DepthMethod, estimate, palm_geometry

FPS = 30.0
NOISE_PX = 4.0
SEED = 2026


# ----------------------------------------------------------------- smoothing
Trajectory = Callable[[float], tuple[float, float]]


def _circle(t: float) -> tuple[float, float]:
    return 320 + 150 * math.cos(1.6 * t), 240 + 150 * math.sin(1.6 * t)


def _figure_eight(t: float) -> tuple[float, float]:
    return 320 + 180 * math.sin(1.2 * t), 240 + 110 * math.sin(2.4 * t)


def _hold_then_stroke(t: float) -> tuple[float, float]:
    """Pointer held still (where jitter is most visible), then a fast straight stroke."""
    if t < 1.5:
        return 200.0, 240.0
    progress = min(1.0, (t - 1.5) / 0.6)
    return 200.0 + 260.0 * progress, 240.0


TRAJECTORIES: dict[str, Trajectory] = {
    "circle": _circle,
    "figure_eight": _figure_eight,
    "hold_then_stroke": _hold_then_stroke,
}


@dataclass(frozen=True)
class SmoothingResult:
    trajectory: str
    filter: str
    rmse_px: float
    jitter_px: float
    still_jitter_px: float


def evaluate_smoothing(
    filters: tuple[str, ...] = ("raw", "exponential", "kalman"),
    seconds: float = 4.0,
    noise_px: float = NOISE_PX,
    seed: int = SEED,
) -> list[SmoothingResult]:
    """RMSE against the true path, and jitter (mean |second difference|) of the output."""
    results = []
    frames = int(seconds * FPS)
    for name, path in TRAJECTORIES.items():
        rng = random.Random(seed)
        times = [i / FPS for i in range(frames)]
        truth = [path(t) for t in times]
        noisy = [(x + rng.gauss(0, noise_px), y + rng.gauss(0, noise_px)) for x, y in truth]
        for kind in filters:
            smoother = make_point_filter(kind)
            output = [
                smoother.update_float(point, now=t) for point, t in zip(noisy, times, strict=True)
            ]
            errors = [math.dist(o, p) for o, p in zip(output, truth, strict=True)]
            second = [
                math.hypot(
                    output[i + 1][0] - 2 * output[i][0] + output[i - 1][0],
                    output[i + 1][1] - 2 * output[i][1] + output[i - 1][1],
                )
                for i in range(1, len(output) - 1)
            ]
            still = (
                [math.dist(output[i], truth[i]) for i, t in enumerate(times) if 0.5 <= t < 1.5]
                if name == "hold_then_stroke"
                else []
            )
            results.append(
                SmoothingResult(
                    trajectory=name,
                    filter=kind,
                    rmse_px=math.sqrt(statistics.mean(e * e for e in errors)),
                    jitter_px=statistics.mean(second),
                    still_jitter_px=statistics.mean(still) if still else float("nan"),
                )
            )
    return results


# --------------------------------------------------------------------- depth
SCENARIOS: dict[str, list[HandPose]] = {
    "translation": [HandPose(distance_cm=d) for d in range(35, 81, 5)],
    "roll": [HandPose(roll=a) for a in range(0, 61, 10)],
    "pitch": [HandPose(pitch=a) for a in range(0, 51, 10)],
    "roll_and_translation": [
        HandPose(distance_cm=50 + d, roll=a)
        for d, a in zip(range(0, 31, 5), range(0, 61, 10), strict=True)
    ],
}


@dataclass(frozen=True)
class DepthResult:
    scenario: str
    method: str
    mean_abs_error: float
    max_abs_error: float
    rotation_detected: float


def evaluate_depth(noise_px: float = 1.5, repeats: int = 20, seed: int = SEED) -> list[DepthResult]:
    """Error of the relative distance against the true one, per scenario and method."""
    rng = random.Random(seed)
    reference = palm_geometry(project(HandPose()), IMAGE_ASPECT)
    results = []
    for scenario, poses in SCENARIOS.items():
        for method in DepthMethod:
            errors, rotated = [], []
            for pose in poses:
                truth = pose.distance_cm / 50.0
                for _ in range(repeats):
                    geometry = palm_geometry(project(pose, noise_px, rng), IMAGE_ASPECT)
                    reading = estimate(geometry, reference, method)
                    errors.append(abs(reading.relative_distance - truth) / truth)
                    rotated.append(reading.motion.value == "rotation")
            results.append(
                DepthResult(
                    scenario,
                    method.value,
                    statistics.mean(errors),
                    max(errors),
                    sum(rotated) / len(rotated),
                )
            )
    return results


# -------------------------------------------------------------------- report
def markdown_report() -> str:
    lines = [
        "## Pointer filters (synthetic trajectories, 30 fps, noise σ = 4 px)",
        "",
        "| Trajectory | Filter | RMSE (px) | Jitter (px) | Still error (px) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in evaluate_smoothing():
        still = "—" if math.isnan(r.still_jitter_px) else f"{r.still_jitter_px:.2f}"
        lines.append(
            f"| {r.trajectory} | {r.filter} | {r.rmse_px:.2f} | {r.jitter_px:.2f} | {still} |"
        )
    lines += [
        "",
        "## Depth estimators (synthetic hand, pinhole camera, noise σ = 1.5 px)",
        "",
        "| Scenario | Method | Mean error | Max error | Classified as rotation |",
        "|---|---|---:|---:|---:|",
    ]
    for r in evaluate_depth():
        lines.append(
            f"| {r.scenario} | {r.method} | {r.mean_abs_error:.1%} | {r.max_abs_error:.1%} "
            f"| {r.rotation_detected:.0%} |"
        )
    return "\n".join(lines)
