"""Build privacy-filtered report data independently of output formats."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from spectra.metrics.tracking_quality import TrackingQuality
from spectra.storage.db import Database
from spectra.storage.models import AuditAction, MetricSummary
from spectra.storage.security import mask_cpf


@dataclass(frozen=True)
class ReportPatient:
    display_name: str
    masked_cpf: str
    age: int | None
    sex_key: str
    dominant_hand: str
    affected_hand: str
    diagnosis_cid: str
    demo_mode: bool


@dataclass(frozen=True)
class MetricRow:
    label_key: str
    value: str
    estimate: bool = False
    label_args: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonRow:
    label_key: str
    current: float
    previous: float | None = None

    @property
    def delta(self) -> float | None:
        return None if self.previous is None else self.current - self.previous

    @property
    def trend_key(self) -> str:
        delta = self.delta
        if delta is None:
            return "report.trend.new"
        if math.isclose(delta, 0.0, abs_tol=1e-9):
            return "report.trend.stable"
        return "report.trend.up" if delta > 0 else "report.trend.down"


@dataclass(frozen=True)
class TrendPoint:
    session_id: str
    started_at: float
    value: float


@dataclass(frozen=True)
class ReportExercise:
    name: str
    kind: str
    reps: int
    target: int
    duration_seconds: float
    #: Title of a therapist-defined exercise; built-in ones are translated by key.
    title: str = ""

    def label(self, catalog=None) -> str:
        from spectra.i18n import active_catalog

        c = catalog or active_catalog()
        if self.title:
            return self.title
        key = "physio.exercise." + self.name
        return c.t(key) if key in c else c.t("report.exercise.guided")

    @property
    def pace_rpm(self) -> float:
        return self.reps * 60 / self.duration_seconds if self.duration_seconds else 0.0


@dataclass
class SessionReport:
    patient: ReportPatient
    session_id: str
    started_at: float
    duration_seconds: float
    hand: str
    gesture_profile: str
    difficulty: str
    exercises: list[ReportExercise] = field(default_factory=list)
    metrics: list[MetricRow] = field(default_factory=list)
    comparison: list[ComparisonRow] = field(default_factory=list)
    trends: dict[str, list[TrendPoint]] = field(default_factory=dict)
    rom: list[tuple[str, float]] = field(default_factory=list)
    pain_before: int | None = None
    pain_after: int | None = None
    therapist_notes: str = ""
    reliable: bool = False
    warning_keys: tuple[str, ...] = ()
    session_number: int = 1
    total_sessions: int = 1

    @property
    def has_estimates(self) -> bool:
        return any(row.estimate for row in self.metrics)

    @property
    def filename_stem(self) -> str:
        return f"sessao_{self.session_id}"


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _format(value: object, percent: bool = False, unit: str = "") -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number * 100 if percent else number:.1f}{'%' if percent else unit}"


def _summary_values(summaries: list[MetricSummary]) -> dict[str, float]:
    # Whole-session summaries take precedence over per-exercise windows.
    chosen = {summary.metric: summary.payload for summary in summaries}
    chosen.update({s.metric: s.payload for s in summaries if s.exercise_run_id is None})
    result: dict[str, float] = {}
    spans = [
        value
        for finger in chosen.get("rom", {}).get("fingers", [])
        if (value := _number(finger.get("span"))) is not None
    ]
    if spans:
        result["report.compare.mean_rom"] = sum(spans) / len(spans)
    score = _number(chosen.get("accuracy", {}).get("best_score"))
    if score is not None:
        result["report.compare.score"] = score
    fatigue = chosen.get("fatigue", {})
    decline = _number(fatigue.get("amplitude_decline"))
    if decline is not None and fatigue.get("measurable", True):
        result["report.compare.amplitude_decline"] = decline * 100
    return result


def _metric_rows(summaries: list[MetricSummary]) -> list[MetricRow]:
    from spectra.i18n import t

    rows = []
    for summary in summaries:
        payload = summary.payload
        if summary.metric in ("rom", "opposition"):
            for finger in payload.get("fingers", []):
                name = str(finger.get("finger", ""))
                angular = summary.metric == "rom"
                rows.append(
                    MetricRow(
                        "report.metric.rom_finger" if angular else "report.metric.opposition",
                        _format(
                            finger.get("span" if angular else "min_distance"),
                            unit="°" if angular else "",
                        ),
                        estimate=angular,
                        label_args={"finger": t(f"finger.{name}")},
                    )
                )
        fields = {
            "wrist": [
                ("rotation_span", "wrist_rotation", False, True, "°"),
                ("openness_span", "wrist_openness", False, True, "°"),
            ],
            "tremor": [
                ("band_power_ratio", "tremor_band", True, False, ""),
                ("mean_jerk", "jerk", False, False, ""),
            ],
            "fatigue": [
                ("amplitude_decline", "amplitude_decline", True, False, ""),
                ("pace_decline", "pace_decline", True, False, ""),
            ],
            "accuracy": [
                ("best_score", "score", False, False, ""),
                ("mean_deviation", "mean_deviation", False, False, ""),
                ("mean_completion", "completion", True, False, ""),
            ],
        }
        for key, label, percent, estimate, unit in fields.get(summary.metric, []):
            value = payload.get(key)
            if (key == "band_power_ratio" and not payload.get("band_measurable", False)) or (
                summary.metric == "fatigue" and not payload.get("measurable", True)
            ):
                value = None
            if key in payload or key == "band_power_ratio":
                rows.append(
                    MetricRow(f"report.metric.{label}", _format(value, percent, unit), estimate)
                )
    return rows


def build_session_report(
    db: Database,
    session_id: str,
    demo_mode: bool = False,
    actor_id: str | None = None,
) -> SessionReport:
    """Read a session, mask identifiers once, and audit report generation.

    Renderers never receive a raw patient record, CPF, or free-form run payload.
    Historical series stop at the selected session, not at the latest database entry.
    """
    session = db.get_session(session_id)
    if session is None:
        raise ValueError("Unknown session")
    patient = db.get_patient(session.patient_id, actor_id=actor_id)
    if patient is None:
        raise ValueError("Unknown patient")
    masked = demo_mode or session.demo_mode
    sessions = db.list_sessions(patient.id)
    index = next(i for i, item in enumerate(sessions) if item.id == session_id)
    history = sessions[: index + 1]
    summaries = db.list_metrics(session_id)
    quality_data = session.tracking_quality
    if not quality_data:
        quality_data = next((s.payload for s in summaries if s.metric == "quality"), {})
    quality = TrackingQuality(
        frames_total=quality_data.get("frames_total", 0),
        frames_with_hand=quality_data.get("frames_with_hand", 0),
        mean_confidence=quality_data.get("mean_confidence", 0.0),
    )
    report = SessionReport(
        patient=ReportPatient(
            patient.display_name(masked),
            "***.***.***-**" if masked else mask_cpf(patient.cpf),
            patient.age_at(datetime.fromtimestamp(session.started_at).date()),
            f"patient.sex.{patient.sex.value}",
            patient.dominant_hand.value,
            patient.affected_hand.value,
            "" if masked else patient.diagnosis_cid,
            masked,
        ),
        session_id=session.id,
        started_at=session.started_at,
        duration_seconds=session.duration,
        hand=session.hand.value,
        gesture_profile=session.gesture_profile_name,
        difficulty=session.difficulty,
        exercises=[
            ReportExercise(
                run.name,
                run.kind,
                run.reps,
                run.target,
                run.duration,
                str(run.payload.get("title") or ""),
            )
            for run in db.list_exercise_runs(session_id)
        ],
        metrics=_metric_rows(summaries),
        pain_before=session.pain_before,
        pain_after=session.pain_after,
        therapist_notes="" if masked else session.notes,
        reliable=quality.reliable,
        warning_keys=(quality.warning_key,) if quality.warning_key else (),
        session_number=index + 1,
        total_sessions=len(sessions),
    )
    for summary in summaries:
        if summary.metric == "rom":
            for finger in summary.payload.get("fingers", []):
                if (span := _number(finger.get("span"))) is not None:
                    report.rom.append((str(finger.get("finger", "")), span))
    previous = _summary_values(db.list_metrics(history[-2].id)) if index else {}
    current = _summary_values(summaries)
    report.comparison = [
        ComparisonRow(key, value, previous.get(key)) for key, value in current.items()
    ]
    for item in history:
        values = current if item.id == session_id else _summary_values(db.list_metrics(item.id))
        for key, value in values.items():
            report.trends.setdefault(key, []).append(TrendPoint(item.id, item.started_at, value))
    db.audit.record(AuditAction.REPORT_GENERATED, subject_id=patient.id, actor_id=actor_id)
    return report
