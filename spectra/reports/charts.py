"""Offline charts returned as PNG bytes; no patient-derived temporary files."""

from __future__ import annotations

import unicodedata
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from spectra.i18n import t
from spectra.reports.builder import SessionReport


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _figure(title_key: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(7, 3.2), layout="constrained")
    axes = figure.subplots()
    axes.set_title(_ascii(t(title_key)))
    return figure, axes


def _png(figure: Figure) -> bytes:
    output = BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    return output.getvalue()


def rom_chart(report: SessionReport) -> bytes | None:
    if not report.rom:
        return None
    figure, axes = _figure("report.chart.rom_title")
    labels = [_ascii(t("finger." + name)) for name, _ in report.rom]
    axes.bar(range(len(labels)), [span for _, span in report.rom], tick_label=labels)
    axes.set_ylabel(_ascii(t("report.chart.degrees") + " (" + t("common.estimate") + ")"))
    return _png(figure)


def exercises_chart(report: SessionReport) -> bytes | None:
    if not report.exercises:
        return None
    figure, axes = _figure("report.chart.exercises_title")
    labels = [_ascii(run.label()) for run in report.exercises]
    positions = list(range(len(labels)))
    axes.bar(
        [x - 0.2 for x in positions],
        [run.reps for run in report.exercises],
        width=0.4,
        label=_ascii(t("report.field.reps")),
    )
    axes.bar(
        [x + 0.2 for x in positions],
        [run.target for run in report.exercises],
        width=0.4,
        label=_ascii(t("report.chart.target")),
    )
    axes.set_xticks(positions, labels)
    axes.legend()
    return _png(figure)


def all_charts(report: SessionReport) -> list[tuple[str, bytes]]:
    charts = []
    for key, image in (
        ("report.chart.rom_title", rom_chart(report)),
        ("report.chart.exercises_title", exercises_chart(report)),
    ):
        if image is not None:
            charts.append((key, image))
    for key, points in report.trends.items():
        if len(points) < 2:
            continue
        figure, axes = _figure(key)
        axes.plot(range(1, len(points) + 1), [point.value for point in points], marker="o")
        axes.set_xlabel(_ascii(t("report.field.session_number")))
        axes.set_xticks(range(1, len(points) + 1))
        charts.append((key, _png(figure)))
    return charts
