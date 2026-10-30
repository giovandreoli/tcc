"""Localized UTF-8 session reports, with no renderer-specific privacy rules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from spectra.i18n import Catalog, active_catalog
from spectra.reports.builder import SessionReport


def sections(report: SessionReport, catalog: Catalog | None = None) -> list[tuple[str, list[str]]]:
    """Shared textual content for TXT and PDF; keys never serve as labels."""
    c = catalog or active_catalog()
    t = c.t

    def value(item: object) -> str:
        return "—" if item is None or item == "" else str(item)

    def row(key: str, item: object) -> str:
        return f"{t('report.field.' + key)}: {value(item)}"

    def hand(item: str) -> str:
        return t("report.hand." + item.lower())

    p = report.patient
    identification = [
        row("patient", p.display_name),
        row("cpf", p.masked_cpf),
        row("age", p.age),
        row("sex", t(p.sex_key)),
        row("dominant_hand", hand(p.dominant_hand)),
        row("affected_hand", hand(p.affected_hand)),
        row("diagnosis", p.diagnosis_cid),
    ]
    session = [
        row("date", datetime.fromtimestamp(report.started_at).strftime("%d/%m/%Y %H:%M")),
        row("duration", f"{report.duration_seconds:.0f} s"),
        row("hand", hand(report.hand)),
        row("gesture_profile", t("gesture.profile." + report.gesture_profile)),
        row("difficulty", t("guided.difficulty." + report.difficulty)),
        row("session_number", f"{report.session_number}/{report.total_sessions}"),
        row("pain_before", report.pain_before),
        row("pain_after", report.pain_after),
    ]
    exercises = []
    for exercise in report.exercises:
        name = exercise.label(c)
        exercises.append(
            f"{name}: {t('report.field.reps')} {exercise.reps}/{exercise.target}; "
            f"{t('report.field.duration')} {exercise.duration_seconds:.0f} s; "
            f"{t('report.field.pace')} {exercise.pace_rpm:.1f}"
        )
    metrics = [
        f"{t(metric.label_key, **metric.label_args)}: {metric.value}"
        + (f" ({t('common.estimate')})" if metric.estimate else "")
        for metric in report.metrics
    ]

    def number(item: float | None) -> str:
        return "—" if item is None else f"{item:.1f}"

    comparison = [
        f"{t(item.label_key)}: {t('report.field.current')} {item.current:.1f}; "
        f"{t('report.field.previous')} {number(item.previous)}; "
        f"{t('report.field.delta')} {value(None if item.delta is None else f'{item.delta:+.1f}')} "
        f"({t(item.trend_key)})"
        for item in report.comparison
    ]
    trend = [
        f"{t(key)}: "
        + "; ".join(
            f"{datetime.fromtimestamp(point.started_at):%d/%m/%Y}: {point.value:.1f}"
            for point in points
        )
        for key, points in report.trends.items()
        if len(points) >= 2
    ]
    result = [
        ("report.section.identification", identification),
        ("report.section.session", session),
    ]
    if report.warning_keys:
        result.append(("report.section.warning", [t(key) for key in report.warning_keys]))
    result.extend(
        [
            ("report.section.exercises", exercises),
            ("report.section.metrics", metrics),
            ("report.section.comparison", comparison),
            ("report.section.trend", trend),
            ("report.section.notes", [report.therapist_notes or t("report.no_notes")]),
        ]
    )
    return result


def render(report: SessionReport, catalog: Catalog | None = None) -> str:
    c = catalog or active_catalog()
    lines = [c.t("report.title"), ""]
    if report.patient.demo_mode:
        lines.extend([c.t("report.demo_mode_notice"), ""])
    for key, content in sections(report, c):
        lines.extend([c.t(key).upper(), *content, ""])
    lines.extend([c.t("report.disclaimer"), ""])
    return "\n".join(lines)


def write(report: SessionReport, directory: Path, catalog: Catalog | None = None) -> Path:
    path = Path(directory) / f"{report.filename_stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(report, catalog), encoding="utf-8")
    return path
