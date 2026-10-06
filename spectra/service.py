"""Application service shared by the therapist panel and the patient app.

The panel is a Streamlit script and is awkward to test; everything it actually *does*
lives here instead, as plain methods over the database, so the behaviour is unit tested
and the UI stays a thin layer of widgets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from spectra.config import AppConfig, load_config
from spectra.consent import ConsentDocument, load_consent
from spectra.gestures.calibration import CalibrationProfile, build_profile
from spectra.gestures.profiles import (
    GestureProfile,
    ProfileIssue,
    get_preset,
    validate_profile,
)
from spectra.guided.scoring import TraceResult
from spectra.metrics.analysis import RunWindow, analyze_run, analyze_session
from spectra.metrics.recorder import SessionRecorder
from spectra.storage.backup import BackupResult, create_backup
from spectra.storage.db import Database
from spectra.storage.models import (
    AuditAction,
    Consent,
    ExerciseRun,
    Hand,
    MetricSummary,
    Patient,
    Session,
    Sex,
)
from spectra.storage.security import is_valid_cpf, is_valid_pin

logger = logging.getLogger(__name__)

MAX_SAMPLES_PERSISTED = 20_000


@dataclass
class PatientForm:
    """Raw form input, validated before it is allowed near the database."""

    name: str = ""
    cpf: str = ""
    birth_date: date | None = None
    sex: Sex = Sex.UNDISCLOSED
    phone: str = ""
    address: str = ""
    dominant_hand: Hand = Hand.RIGHT
    affected_hand: Hand = Hand.RIGHT
    diagnosis_cid: str = ""
    injury_date: date | None = None
    referring_professional: str = ""
    contraindications: str = ""
    notes: str = ""
    pin: str = ""

    def validate(self) -> list[str]:
        """Return i18n keys for every problem found; empty means valid."""
        problems: list[str] = []
        if not self.name.strip():
            problems.append("panel.error.name_required")
        if not is_valid_cpf(self.cpf):
            problems.append("panel.error.cpf_invalid")
        if self.pin and not is_valid_pin(self.pin):
            problems.append("panel.error.pin_invalid")
        if self.birth_date and self.birth_date > date.today():
            problems.append("panel.error.birth_date_future")
        if self.injury_date and self.injury_date > date.today():
            problems.append("panel.error.injury_date_future")
        return problems

    def to_patient(self) -> Patient:
        return Patient(
            name=self.name.strip(),
            cpf=self.cpf,
            birth_date=self.birth_date,
            sex=self.sex,
            phone=self.phone.strip(),
            address=self.address.strip(),
            dominant_hand=self.dominant_hand,
            affected_hand=self.affected_hand,
            diagnosis_cid=self.diagnosis_cid.strip(),
            injury_date=self.injury_date,
            referring_professional=self.referring_professional.strip(),
            contraindications=self.contraindications.strip(),
            notes=self.notes.strip(),
        )


@dataclass
class SessionPlan:
    """What the therapist configured before handing the screen to the patient."""

    patient_id: str
    therapist_id: str | None = None
    hand: Hand = Hand.RIGHT
    gesture_profile_name: str = "standard"
    difficulty: str = "easy"
    duration_minutes: int = 20
    exercises: tuple[str, ...] = ("open_close", "finger_touch", "finger_wave")
    demo_mode: bool = False

    def gesture_profile(self) -> GestureProfile:
        return get_preset(self.gesture_profile_name)

    def profile_warnings(self) -> list[ProfileIssue]:
        return validate_profile(self.gesture_profile())


@dataclass
class SpectraService:
    """Use-case layer over :class:`spectra.storage.db.Database`."""

    db: Database
    config: AppConfig = field(default_factory=load_config)

    @classmethod
    def open(cls, config: AppConfig | None = None) -> SpectraService:
        settings = config or load_config()
        settings.ensure_directories()
        return cls(db=Database.open(settings.db_path, settings.key_path), config=settings)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> SpectraService:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---------------------------------------------------------------- patients
    def create_patient(self, form: PatientForm, actor_id: str | None = None) -> Patient:
        problems = form.validate()
        if problems:
            raise ValueError(problems)
        return self.db.create_patient(form.to_patient(), pin=form.pin or None, actor_id=actor_id)

    def save_calibration(self, patient_id: str, profile: CalibrationProfile) -> None:
        patient = self.db.get_patient(patient_id, audit=False)
        if patient is None:
            raise ValueError("unknown patient")
        patient.calibration = profile.to_dict()
        self.db.update_patient(patient)

    def calibrate(
        self, patient_id: str, open_samples: list, closed_samples: list
    ) -> CalibrationProfile:
        """Build and store a calibration profile from two recorded postures."""
        profile = build_profile(open_samples, closed_samples)
        self.save_calibration(patient_id, profile)
        return profile

    def calibration_for(self, patient_id: str) -> CalibrationProfile:
        patient = self.db.get_patient(patient_id, audit=False)
        if patient is None or not patient.calibration:
            return CalibrationProfile()
        try:
            return CalibrationProfile.from_dict(patient.calibration)
        except (KeyError, TypeError, ValueError):
            logger.warning("stored calibration is unreadable; falling back to defaults")
            return CalibrationProfile()

    # ----------------------------------------------------------------- consent
    def current_consent_document(self) -> ConsentDocument:
        return load_consent(self.config.locale)

    def record_consent(
        self, patient_id: str, signed_by: str, actor_id: str | None = None
    ) -> Consent:
        """Store an acceptance of the *current* consent text."""
        document = self.current_consent_document()
        return self.db.record_consent(
            Consent(
                patient_id=patient_id,
                version=document.version,
                document_hash=document.document_hash,
                signed_by=signed_by,
            ),
            actor_id=actor_id,
        )

    def has_consent(self, patient_id: str) -> bool:
        return self.db.active_consent(patient_id) is not None

    # ---------------------------------------------------------------- sessions
    def start_session(self, plan: SessionPlan) -> Session:
        """Open a session. Raises :class:`ConsentRequiredError` without a consent."""
        return self.db.start_session(
            Session(
                patient_id=plan.patient_id,
                therapist_id=plan.therapist_id,
                hand=plan.hand,
                gesture_profile_name=plan.gesture_profile_name,
                difficulty=plan.difficulty,
                demo_mode=plan.demo_mode,
            ),
            actor_id=plan.therapist_id,
        )

    def finish_session(
        self,
        session: Session,
        recorder: SessionRecorder,
        runs: list[RunWindow] | None = None,
        traces: list[TraceResult] | None = None,
        pain_before: int | None = None,
        pain_after: int | None = None,
        notes: str = "",
        persist_samples: bool = True,
    ) -> Session:
        """Compute every metric, persist runs, metrics and samples, and close the session."""
        runs = runs or []
        quality = recorder.quality()
        session.tracking_quality = quality.to_dict()
        session.pain_before = pain_before
        session.pain_after = pain_after
        session.notes = notes

        for window in runs:
            run = self.db.add_exercise_run(
                ExerciseRun(
                    session_id=session.id,
                    kind=window.kind,
                    name=window.name,
                    started_at=window.started_at,
                    ended_at=window.ended_at,
                    reps=window.reps,
                    target=window.target,
                    payload={"rep_times": list(window.rep_times)},
                )
            )
            for metric, payload in analyze_run(recorder, window).items():
                self.db.add_metric(
                    MetricSummary(
                        session_id=session.id,
                        exercise_run_id=run.id,
                        metric=metric,
                        payload=payload,
                    )
                )

        for metric, payload in analyze_session(recorder, runs, traces or []).items():
            self.db.add_metric(MetricSummary(session_id=session.id, metric=metric, payload=payload))

        if persist_samples and recorder.samples:
            payloads = [sample.to_dict() for sample in recorder.samples[:MAX_SAMPLES_PERSISTED]]
            self.db.add_samples(session.id, payloads)

        return self.db.finish_session(session, actor_id=session.therapist_id)

    # ----------------------------------------------------------------- reports
    def generate_reports(
        self, session_id: str, demo_mode: bool | None = None, actor_id: str | None = None
    ):
        from spectra.reports import generate
        from spectra.reports.builder import build_session_report

        masked = self.config.demo_mode if demo_mode is None else demo_mode
        report = build_session_report(self.db, session_id, demo_mode=masked, actor_id=actor_id)
        return report, generate(report, self.config.reports_dir)

    def export_reports_to_share(self, generated, patient_id: str, actor_id: str | None = None):
        from spectra.reports import export_to_share

        if self.config.share_dir is None:
            raise ValueError("share_dir is not configured")
        return export_to_share(
            generated, self.config.share_dir, db=self.db, patient_id=patient_id, actor_id=actor_id
        )

    # ------------------------------------------------------------------ backup
    def backup(
        self, password: str, destination: Path | None = None, actor_id: str | None = None
    ) -> BackupResult:
        """Write an encrypted backup of the database and the reports."""
        target = (
            Path(destination) if destination else (self.config.share_dir or self.config.exports_dir)
        )
        result = create_backup(
            target,
            password,
            db_path=self.config.db_path,
            extra_dirs=(self.config.reports_dir,),
        )
        self.db.audit.record(
            AuditAction.BACKUP_CREATED, subject_id=None, subject_type="system", actor_id=actor_id
        )
        return result
