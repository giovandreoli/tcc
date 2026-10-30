"""Application service shared by the therapist panel and the patient app.

The panel is a Streamlit script and is awkward to test; everything it actually *does*
lives here instead, as plain methods over the database, so the behaviour is unit tested
and the UI stays a thin layer of widgets.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from spectra.config import AppConfig, load_config
from spectra.consent import ConsentDocument, load_consent
from spectra.exercises.catalog import (
    BUILTIN_SPECS,
    CUSTOM_PREFIX,
    ExerciseSpec,
)
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
    PatientDocument,
    Session,
    Sex,
)
from spectra.storage.security import is_valid_cpf, is_valid_pin

logger = logging.getLogger(__name__)

MAX_SAMPLES_PERSISTED = 20_000

DOCUMENT_CATEGORIES = ("physio_report", "medical_report", "exam", "referral", "other")
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_REPORT_CHARS = 20_000
#: Accepted uploads, identified by content rather than by the (untrusted) file extension.
MEDIA_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "application/pdf": (b"%PDF-",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
}


def detect_media_type(content: bytes) -> str | None:
    for media_type, signatures in MEDIA_SIGNATURES.items():
        if any(content.startswith(signature) for signature in signatures):
            return media_type
    return None


@dataclass
class DocumentForm:
    """A clinical document to add: a written report, an uploaded file, or both."""

    category: str = "physio_report"
    title: str = ""
    issued_on: date | None = None
    body: str = ""
    filename: str = ""
    content: bytes | None = None

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.category not in DOCUMENT_CATEGORIES:
            problems.append("document.error.category")
        if not 3 <= len(self.title.strip()) <= 120:
            problems.append("document.error.title")
        if self.issued_on and self.issued_on > date.today():
            problems.append("document.error.date_future")
        if not self.body.strip() and not self.content:
            problems.append("document.error.empty")
        if len(self.body) > MAX_REPORT_CHARS:
            problems.append("document.error.body_long")
        if self.content is not None:
            if len(self.content) > MAX_DOCUMENT_BYTES:
                problems.append("document.error.too_large")
            elif detect_media_type(self.content) is None:
                problems.append("document.error.type")
        return problems


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

    # --------------------------------------------------------------- learning
    def save_pinch_calibration(self, patient_id: str, thresholds) -> None:
        """Keep the finger calibration and add the patient's own click thresholds."""
        profile = self.calibration_for(patient_id)
        profile.pinch = thresholds
        self.save_calibration(patient_id, profile)

    def save_gesture_model(
        self, patient_id: str, model, bindings: dict[str, str], actor_id: str | None = None
    ) -> None:
        from spectra.domains import Command

        model.validate()
        clean = {label: Command(command).value for label, command in bindings.items() if command}
        self.db.save_gesture_model(patient_id, model.to_dict(), clean, actor_id=actor_id)

    def load_gesture_model(self, patient_id: str):
        from spectra.learning.classifier import GestureClassifier

        stored = self.db.get_gesture_model(patient_id)
        if stored is None:
            return None, {}
        model, bindings = stored
        return GestureClassifier.from_dict(model), bindings

    def export_gesture_model(self, patient_id: str) -> Path:
        """Write the patient's learned gestures where the desktop application loads them."""
        import json

        stored = self.db.get_gesture_model(patient_id)
        if stored is None:
            raise ValueError("no gesture model for this patient")
        model, bindings = stored
        path = self.config.gesture_model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "bindings": bindings}), encoding="utf-8")
        return path

    # ------------------------------------------------------- custom exercises
    def custom_exercises(self, patient_id: str) -> list[ExerciseSpec]:
        specs = []
        for exercise_id, data in self.db.list_custom_exercises(patient_id):
            try:
                specs.append(ExerciseSpec.from_dict(data).with_key(CUSTOM_PREFIX + exercise_id))
            except (KeyError, TypeError, ValueError):
                logger.warning("skipping an unreadable custom exercise")
        return specs

    def available_exercises(self, patient_id: str) -> list[ExerciseSpec]:
        """Built-in exercises followed by the patient's own ones."""
        return [*BUILTIN_SPECS.values(), *self.custom_exercises(patient_id)]

    def save_custom_exercise(
        self, patient_id: str, spec: ExerciseSpec, actor_id: str | None = None
    ) -> ExerciseSpec:
        """Validate and store a therapist-defined exercise. Raises ``ValueError`` with keys."""
        spec = replace(
            spec,
            builtin=False,
            title=spec.title.strip(),
            instructions=tuple(line.strip() for line in spec.instructions if line.strip()),
        )
        problems = spec.validate()
        if problems:
            raise ValueError(problems)
        if self.db.get_patient(patient_id, audit=False) is None:
            raise ValueError(["exercise.error.patient"])
        exercise_id = self.db.add_custom_exercise(
            patient_id, spec.with_key("").to_dict(), actor_id=actor_id
        )
        return spec.with_key(CUSTOM_PREFIX + exercise_id)

    def archive_custom_exercise(
        self, patient_id: str, key: str, actor_id: str | None = None
    ) -> bool:
        if not key.startswith(CUSTOM_PREFIX):
            return False
        return self.db.archive_custom_exercise(
            patient_id, key.removeprefix(CUSTOM_PREFIX), actor_id=actor_id
        )

    def resolve_exercises(self, patient_id: str, keys: tuple[str, ...]) -> tuple[ExerciseSpec, ...]:
        """Turn the keys chosen for a session into specs; unknown keys raise ``ValueError``."""
        available = {spec.key: spec for spec in self.available_exercises(patient_id)}
        unknown = [key for key in keys if key not in available]
        if unknown or not keys:
            raise ValueError("unknown exercise")
        return tuple(available[key] for key in keys)

    # -------------------------------------------------------------- documents
    def add_document(
        self, patient_id: str, form: DocumentForm, actor_id: str | None = None
    ) -> PatientDocument:
        """Store a written report and/or an uploaded file. Raises ``ValueError`` with keys."""
        problems = form.validate()
        if problems:
            raise ValueError(problems)
        if self.db.get_patient(patient_id, audit=False) is None:
            raise ValueError(["document.error.patient"])
        content = form.content
        document = PatientDocument(
            patient_id=patient_id,
            therapist_id=actor_id,
            category=form.category,
            title=form.title.strip(),
            issued_on=form.issued_on,
            body=form.body.strip(),
            filename=Path(form.filename).name[:200] if content else "",
            media_type=(detect_media_type(content) or "") if content else "",
            size=len(content) if content else 0,
            sha256=hashlib.sha256(content).hexdigest() if content else "",
        )
        return self.db.add_document(document, content, actor_id=actor_id)

    def written_report_pdf(self, document_id: str, actor_id: str | None = None) -> bytes:
        """Render a written report as a PDF for printing or sharing with the patient."""
        from spectra.reports.document import written_report_pdf

        document = self.db.get_document(document_id)
        if document is None or not document.body:
            raise ValueError("no written report")
        patient = self.db.get_patient(document.patient_id, audit=False)
        therapist = self.db.get_therapist(document.therapist_id) if document.therapist_id else None
        self.db.audit.record(
            AuditAction.DOCUMENT_VIEWED, subject_id=document.patient_id, actor_id=actor_id
        )
        return written_report_pdf(
            document,
            patient_name=patient.name if patient else "",
            therapist_name=therapist.name if therapist else "",
        )

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
                    payload={
                        "rep_times": list(window.rep_times),
                        **({"title": window.title} if window.title else {}),
                        **({"details": window.details} if window.details else {}),
                    },
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
            extra_dirs=(self.config.reports_dir, self.config.documents_dir),
        )
        self.db.audit.record(
            AuditAction.BACKUP_CREATED, subject_id=None, subject_type="system", actor_id=actor_id
        )
        return result
