"""Turn a session recording into the metric payloads that get persisted.

This is the seam between :mod:`spectra.metrics` (pure analysis) and
:mod:`spectra.storage` (persistence). Keeping it as a function of plain data means the
whole analysis can be exercised without a database and without a camera.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from spectra.guided.scoring import TraceResult
from spectra.metrics.accuracy import summarise_traces
from spectra.metrics.fatigue import fatigue_metrics
from spectra.metrics.recorder import SessionRecorder
from spectra.metrics.rom import finger_rom, opposition_rom, wrist_estimate
from spectra.metrics.tremor import tremor_metrics


@dataclass(frozen=True)
class RunWindow:
    """One exercise run reduced to what the analysis needs."""

    name: str
    kind: str = "physio"
    started_at: float = 0.0
    ended_at: float = 0.0
    reps: int = 0
    target: int = 0
    rep_times: tuple[float, ...] = ()

    @property
    def duration(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


def analyze_session(
    recorder: SessionRecorder,
    runs: Sequence[RunWindow] = (),
    traces: Sequence[TraceResult] = (),
) -> dict[str, dict]:
    """Return ``{metric_name: payload}`` for the whole session.

    Payloads are plain dictionaries so new metrics never require a schema migration.
    """
    samples = recorder.samples
    rep_times = [stamp for run in runs for stamp in run.rep_times]

    metrics: dict[str, dict] = {
        "quality": recorder.quality().to_dict(),
        "rom": {"fingers": [rom.to_dict() for rom in finger_rom(samples)]},
        "opposition": {"fingers": [rom.to_dict() for rom in opposition_rom(samples)]},
        "wrist": wrist_estimate(samples).to_dict(),
        "tremor": tremor_metrics(recorder.trajectory).to_dict(),
        "fatigue": fatigue_metrics(samples, rep_times).to_dict(),
    }
    if traces:
        metrics["accuracy"] = summarise_traces(traces).to_dict()
    return metrics


def analyze_run(recorder: SessionRecorder, run: RunWindow) -> dict[str, dict]:
    """Per-exercise metrics, sliced from the session recording by time window."""
    samples = recorder.window(run.started_at, run.ended_at)
    trajectory = recorder.trajectory_window(run.started_at, run.ended_at)
    return {
        "rom": {"fingers": [rom.to_dict() for rom in finger_rom(samples)]},
        "tremor": tremor_metrics(trajectory).to_dict(),
        "fatigue": fatigue_metrics(samples, list(run.rep_times)).to_dict(),
    }
