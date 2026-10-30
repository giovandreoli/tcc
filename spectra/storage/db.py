"""SQLite persistence.

The database lives on local disk inside the data directory, never inside a cloud-synced
folder (SQLite and file sync corrupt each other). Backups and reports are what gets
exported to the shared folder.

Invariants enforced here rather than by convention:

* a session cannot be created for a patient without an active consent;
* sensitive columns are encrypted on write and decrypted on read, driven by
  :data:`spectra.storage.models.ENCRYPTED_FIELDS`;
* every read, export and deletion of patient data writes an audit entry;
* authentication failures count towards a lockout.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from spectra.storage.audit import AuditLog
from spectra.storage.errors import StorageError
from spectra.storage.models import (
    AuditAction,
    Consent,
    ExerciseRun,
    Hand,
    MetricSummary,
    Patient,
    PatientDocument,
    Sample,
    Session,
    Sex,
    Therapist,
    new_id,
    utc_now,
)
from spectra.storage.security import (
    Cipher,
    KeyStore,
    LockoutState,
    hash_secret,
    is_locked,
    is_valid_pin,
    next_lockout,
    validate_cpf,
    verify_secret,
)

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS therapist (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT,
    created_at REAL NOT NULL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until REAL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS patient (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    cpf_enc TEXT,
    cpf_hash TEXT UNIQUE,
    birth_date TEXT,
    sex TEXT NOT NULL DEFAULT 'undisclosed',
    phone_enc TEXT,
    address_enc TEXT,
    dominant_hand TEXT NOT NULL DEFAULT 'Right',
    affected_hand TEXT NOT NULL DEFAULT 'Right',
    diagnosis_cid TEXT,
    injury_date TEXT,
    referring_professional TEXT,
    contraindications_enc TEXT,
    notes_enc TEXT,
    gesture_profile TEXT NOT NULL DEFAULT '{}',
    calibration TEXT NOT NULL DEFAULT '{}',
    pin_hash TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    anonymized_at REAL
);

CREATE TABLE IF NOT EXISTS consent (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    accepted_at REAL NOT NULL,
    document_hash TEXT NOT NULL,
    signed_by TEXT,
    revoked_at REAL
);

CREATE TABLE IF NOT EXISTS session (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    therapist_id TEXT REFERENCES therapist(id),
    consent_id TEXT REFERENCES consent(id),
    started_at REAL NOT NULL,
    ended_at REAL,
    hand TEXT NOT NULL DEFAULT 'Right',
    gesture_profile_name TEXT NOT NULL DEFAULT 'standard',
    difficulty TEXT NOT NULL DEFAULT 'easy',
    tracking_quality TEXT NOT NULL DEFAULT '{}',
    pain_before INTEGER,
    pain_after INTEGER,
    notes_enc TEXT,
    demo_mode INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS exercise_run (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    reps INTEGER NOT NULL DEFAULT 0,
    target INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS metric_summary (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    exercise_run_id TEXT REFERENCES exercise_run(id) ON DELETE CASCADE,
    metric TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sample (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    at REAL NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    at REAL NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    action TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_session_patient ON session(patient_id, started_at);
CREATE INDEX IF NOT EXISTS idx_run_session ON exercise_run(session_id);
CREATE INDEX IF NOT EXISTS idx_metric_session ON metric_summary(session_id);
CREATE INDEX IF NOT EXISTS idx_sample_session ON sample(session_id, at);
CREATE INDEX IF NOT EXISTS idx_audit_subject ON audit_log(subject_id, at);
CREATE INDEX IF NOT EXISTS idx_consent_patient ON consent(patient_id, accepted_at);

CREATE TABLE IF NOT EXISTS browser_room (
    code_hash TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE REFERENCES session(id) ON DELETE CASCADE,
    expires_at REAL NOT NULL,
    duration_minutes INTEGER NOT NULL,
    exercises TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'waiting',
    claim_hash TEXT,
    heartbeat_at REAL
);

CREATE TABLE IF NOT EXISTS custom_exercise (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    therapist_id TEXT REFERENCES therapist(id),
    spec TEXT NOT NULL,
    created_at REAL NOT NULL,
    archived_at REAL
);

CREATE TABLE IF NOT EXISTS patient_document (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    therapist_id TEXT REFERENCES therapist(id),
    category TEXT NOT NULL,
    title_enc TEXT NOT NULL,
    issued_on TEXT,
    body_enc TEXT,
    filename_enc TEXT,
    media_type TEXT,
    size INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exercise_patient ON custom_exercise(patient_id, created_at);
CREATE INDEX IF NOT EXISTS idx_document_patient ON patient_document(patient_id, created_at);

CREATE TABLE IF NOT EXISTS gesture_model (
    patient_id TEXT PRIMARY KEY REFERENCES patient(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    bindings TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS usability_participant (
    code TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    consent INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS usability_result (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL REFERENCES usability_participant(code) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    condition TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
"""


class ConsentRequiredError(StorageError):
    """Raised when a session is started for a patient without an active consent."""


class AccountLockedError(StorageError):
    """Raised when authentication is attempted on a locked account."""


class DuplicateCPFError(StorageError):
    """Raised when a CPF is already registered. Carries no CPF value."""


def _to_iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _from_iso(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


class Database:
    """Repository over SQLite, owning encryption and the audit trail."""

    def __init__(self, path: Path, cipher: Cipher) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = cipher
        self._local = threading.local()
        self._connection.executescript(SCHEMA)
        if not self._connection.execute("SELECT 1 FROM schema_info").fetchone():
            self._connection.execute(
                "INSERT INTO schema_info (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        else:
            # New tables are created by the idempotent schema script; only the marker moves.
            self._connection.execute(
                "UPDATE schema_info SET version = ? WHERE version < ?",
                (SCHEMA_VERSION, SCHEMA_VERSION),
            )
        self._connection.commit()
        self.audit = AuditLog(lambda: self._connection)
        self.documents_dir = self.path.parent / "documents"

    @property
    def _connection(self) -> sqlite3.Connection:
        """The calling thread's connection, opened on first use.

        Streamlit shares one cached service across script threads and sqlite3 connections
        cannot cross threads, so each thread gets its own connection to the same file.
        """
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(str(self.path))
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection = connection
        return connection

    @classmethod
    def open(cls, db_path: Path, key_path: Path) -> Database:
        """Open (creating if needed) the database with the key store's master key."""
        return cls(db_path, Cipher.from_key_store(KeyStore(Path(key_path))))

    def close(self) -> None:
        """Close the calling thread's connection; other threads' close when they end."""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    # ------------------------------------------------------------- therapists
    def create_therapist(self, name: str, email: str, password: str) -> Therapist:
        therapist = Therapist(
            name=name, email=email.strip().lower(), password_hash=hash_secret(password)
        )
        self._connection.execute(
            "INSERT INTO therapist (id, name, email, password_hash, created_at, failed_attempts,"
            " locked_until, is_active) VALUES (?, ?, ?, ?, ?, 0, NULL, 1)",
            (
                therapist.id,
                therapist.name,
                therapist.email,
                therapist.password_hash,
                therapist.created_at,
            ),
        )
        self._connection.commit()
        return therapist

    def get_therapist(self, therapist_id: str) -> Therapist | None:
        row = self._connection.execute(
            "SELECT * FROM therapist WHERE id = ?", (therapist_id,)
        ).fetchone()
        return self._therapist_from_row(row) if row else None

    def get_therapist_by_email(self, email: str) -> Therapist | None:
        row = self._connection.execute(
            "SELECT * FROM therapist WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return self._therapist_from_row(row) if row else None

    def authenticate_therapist(
        self, email: str, password: str, now: float | None = None
    ) -> LockoutState:
        """Verify a therapist password, applying and recording the lockout policy."""
        moment = utc_now() if now is None else now
        therapist = self.get_therapist_by_email(email)
        if therapist is None:
            return LockoutState(authenticated=False, locked=False, failed_attempts=0)
        if is_locked(therapist.locked_until, moment):
            raise AccountLockedError("therapist account is locked")

        success = verify_secret(password, therapist.password_hash)
        state = next_lockout(success, therapist.failed_attempts, moment)
        self._connection.execute(
            "UPDATE therapist SET failed_attempts = ?, locked_until = ? WHERE id = ?",
            (state.failed_attempts, state.locked_until, therapist.id),
        )
        self._connection.commit()
        self.audit.record(
            AuditAction.LOGIN_SUCCESS if success else AuditAction.LOGIN_FAILURE,
            subject_id=therapist.id,
            subject_type="therapist",
            actor_id=therapist.id,
        )
        if state.locked:
            self.audit.record(
                AuditAction.ACCOUNT_LOCKED, subject_id=therapist.id, subject_type="therapist"
            )
        return state

    @staticmethod
    def _therapist_from_row(row: sqlite3.Row) -> Therapist:
        return Therapist(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=row["created_at"],
            failed_attempts=row["failed_attempts"],
            locked_until=row["locked_until"],
            is_active=bool(row["is_active"]),
        )

    # ---------------------------------------------------------------- patients
    def create_patient(
        self, patient: Patient, pin: str | None = None, actor_id: str | None = None
    ) -> Patient:
        """Insert a patient; the CPF is validated, encrypted and keyed-hashed."""
        cpf_digits = validate_cpf(patient.cpf) if patient.cpf else ""
        cpf_hash = self._cipher.cpf_hash(cpf_digits) if cpf_digits else None
        if (
            cpf_hash
            and self._connection.execute(
                "SELECT 1 FROM patient WHERE cpf_hash = ?", (cpf_hash,)
            ).fetchone()
        ):
            raise DuplicateCPFError("a patient with this CPF already exists")

        if pin is not None:
            if not is_valid_pin(pin):
                raise StorageError("the PIN must be exactly four digits")
            patient.pin_hash = hash_secret(pin)

        patient.updated_at = utc_now()
        self._connection.execute(
            "INSERT INTO patient (id, name, cpf_enc, cpf_hash, birth_date, sex, phone_enc,"
            " address_enc, dominant_hand, affected_hand, diagnosis_cid, injury_date,"
            " referring_professional, contraindications_enc, notes_enc, gesture_profile,"
            " calibration, pin_hash, failed_attempts, locked_until, created_at, updated_at,"
            " anonymized_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0,"
            " NULL, ?, ?, NULL)",
            (
                patient.id,
                patient.name,
                self._cipher.encrypt(cpf_digits),
                cpf_hash,
                _to_iso(patient.birth_date),
                patient.sex.value,
                self._cipher.encrypt(patient.phone),
                self._cipher.encrypt(patient.address),
                patient.dominant_hand.value,
                patient.affected_hand.value,
                patient.diagnosis_cid,
                _to_iso(patient.injury_date),
                patient.referring_professional,
                self._cipher.encrypt(patient.contraindications),
                self._cipher.encrypt(patient.notes),
                json.dumps(patient.gesture_profile),
                json.dumps(patient.calibration),
                patient.pin_hash,
                patient.created_at,
                patient.updated_at,
            ),
        )
        self._connection.commit()
        self.audit.record(AuditAction.PATIENT_CREATED, subject_id=patient.id, actor_id=actor_id)
        return patient

    def update_patient(self, patient: Patient, actor_id: str | None = None) -> Patient:
        patient.updated_at = utc_now()
        cpf_digits = validate_cpf(patient.cpf) if patient.cpf else ""
        self._connection.execute(
            "UPDATE patient SET name = ?, cpf_enc = ?, cpf_hash = ?, birth_date = ?, sex = ?,"
            " phone_enc = ?, address_enc = ?, dominant_hand = ?, affected_hand = ?,"
            " diagnosis_cid = ?, injury_date = ?, referring_professional = ?,"
            " contraindications_enc = ?, notes_enc = ?, gesture_profile = ?, calibration = ?,"
            " updated_at = ? WHERE id = ?",
            (
                patient.name,
                self._cipher.encrypt(cpf_digits),
                self._cipher.cpf_hash(cpf_digits) if cpf_digits else None,
                _to_iso(patient.birth_date),
                patient.sex.value,
                self._cipher.encrypt(patient.phone),
                self._cipher.encrypt(patient.address),
                patient.dominant_hand.value,
                patient.affected_hand.value,
                patient.diagnosis_cid,
                _to_iso(patient.injury_date),
                patient.referring_professional,
                self._cipher.encrypt(patient.contraindications),
                self._cipher.encrypt(patient.notes),
                json.dumps(patient.gesture_profile),
                json.dumps(patient.calibration),
                patient.updated_at,
                patient.id,
            ),
        )
        self._connection.commit()
        self.audit.record(AuditAction.PATIENT_UPDATED, subject_id=patient.id, actor_id=actor_id)
        return patient

    def get_patient(
        self, patient_id: str, actor_id: str | None = None, audit: bool = True
    ) -> Patient | None:
        row = self._connection.execute(
            "SELECT * FROM patient WHERE id = ?", (patient_id,)
        ).fetchone()
        if row is None:
            return None
        if audit:
            self.audit.record(AuditAction.PATIENT_VIEWED, subject_id=patient_id, actor_id=actor_id)
        return self._patient_from_row(row)

    def find_patient_by_cpf(self, cpf: str, actor_id: str | None = None) -> Patient | None:
        """Look a patient up by CPF through the keyed hash; the CPF is never queried in clear."""
        digits = validate_cpf(cpf)
        row = self._connection.execute(
            "SELECT * FROM patient WHERE cpf_hash = ?", (self._cipher.cpf_hash(digits),)
        ).fetchone()
        if row is None:
            return None
        self.audit.record(AuditAction.PATIENT_VIEWED, subject_id=row["id"], actor_id=actor_id)
        return self._patient_from_row(row)

    def list_patients(self) -> list[Patient]:
        rows = self._connection.execute("SELECT * FROM patient ORDER BY name").fetchall()
        return [self._patient_from_row(row) for row in rows]

    def set_patient_pin(self, patient_id: str, pin: str) -> None:
        if not is_valid_pin(pin):
            raise StorageError("the PIN must be exactly four digits")
        self._connection.execute(
            "UPDATE patient SET pin_hash = ?, failed_attempts = 0, locked_until = NULL"
            " WHERE id = ?",
            (hash_secret(pin), patient_id),
        )
        self._connection.commit()

    def authenticate_patient(
        self, patient_id: str, pin: str, now: float | None = None
    ) -> LockoutState:
        moment = utc_now() if now is None else now
        patient = self.get_patient(patient_id, audit=False)
        if patient is None:
            return LockoutState(authenticated=False, locked=False, failed_attempts=0)
        if is_locked(patient.locked_until, moment):
            raise AccountLockedError("patient account is locked")

        success = verify_secret(pin, patient.pin_hash)
        state = next_lockout(success, patient.failed_attempts, moment)
        self._connection.execute(
            "UPDATE patient SET failed_attempts = ?, locked_until = ? WHERE id = ?",
            (state.failed_attempts, state.locked_until, patient.id),
        )
        self._connection.commit()
        self.audit.record(
            AuditAction.LOGIN_SUCCESS if success else AuditAction.LOGIN_FAILURE,
            subject_id=patient.id,
            actor_type="patient",
            actor_id=patient.id,
        )
        if state.locked:
            self.audit.record(
                AuditAction.ACCOUNT_LOCKED, subject_id=patient.id, actor_type="patient"
            )
        return state

    def _patient_from_row(self, row: sqlite3.Row) -> Patient:
        return Patient(
            id=row["id"],
            name=row["name"],
            cpf=self._cipher.decrypt(row["cpf_enc"]) or "",
            birth_date=_from_iso(row["birth_date"]),
            sex=Sex(row["sex"]),
            phone=self._cipher.decrypt(row["phone_enc"]) or "",
            address=self._cipher.decrypt(row["address_enc"]) or "",
            dominant_hand=Hand(row["dominant_hand"]),
            affected_hand=Hand(row["affected_hand"]),
            diagnosis_cid=row["diagnosis_cid"] or "",
            injury_date=_from_iso(row["injury_date"]),
            referring_professional=row["referring_professional"] or "",
            contraindications=self._cipher.decrypt(row["contraindications_enc"]) or "",
            notes=self._cipher.decrypt(row["notes_enc"]) or "",
            gesture_profile=json.loads(row["gesture_profile"] or "{}"),
            calibration=json.loads(row["calibration"] or "{}"),
            pin_hash=row["pin_hash"],
            failed_attempts=row["failed_attempts"],
            locked_until=row["locked_until"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            anonymized_at=row["anonymized_at"],
        )

    # ----------------------------------------------------------------- consent
    def record_consent(self, consent: Consent, actor_id: str | None = None) -> Consent:
        self._connection.execute(
            "INSERT INTO consent (id, patient_id, version, accepted_at, document_hash, signed_by,"
            " revoked_at) VALUES (?, ?, ?, ?, ?, ?, NULL)",
            (
                consent.id,
                consent.patient_id,
                consent.version,
                consent.accepted_at,
                consent.document_hash,
                consent.signed_by,
            ),
        )
        self._connection.commit()
        self.audit.record(
            AuditAction.CONSENT_RECORDED,
            subject_id=consent.patient_id,
            actor_id=actor_id,
            detail=consent.version,
        )
        return consent

    def active_consent(self, patient_id: str) -> Consent | None:
        row = self._connection.execute(
            "SELECT * FROM consent WHERE patient_id = ? AND revoked_at IS NULL"
            " ORDER BY accepted_at DESC LIMIT 1",
            (patient_id,),
        ).fetchone()
        if row is None:
            return None
        return Consent(
            id=row["id"],
            patient_id=row["patient_id"],
            version=row["version"],
            accepted_at=row["accepted_at"],
            document_hash=row["document_hash"],
            signed_by=row["signed_by"] or "",
            revoked_at=row["revoked_at"],
        )

    def revoke_consent(self, consent_id: str, actor_id: str | None = None) -> None:
        row = self._connection.execute(
            "SELECT patient_id FROM consent WHERE id = ?", (consent_id,)
        ).fetchone()
        self._connection.execute(
            "UPDATE consent SET revoked_at = ? WHERE id = ?", (utc_now(), consent_id)
        )
        self._connection.commit()
        if row:
            self.audit.record(
                AuditAction.CONSENT_REVOKED, subject_id=row["patient_id"], actor_id=actor_id
            )

    # ---------------------------------------------------------------- sessions
    def start_session(self, session: Session, actor_id: str | None = None) -> Session:
        """Create a session. **Refuses** to do so without an active consent."""
        consent = self.active_consent(session.patient_id)
        if consent is None:
            raise ConsentRequiredError("the patient has no active consent (TCLE) on record")
        session.consent_id = consent.id
        self._connection.execute(
            "INSERT INTO session (id, patient_id, therapist_id, consent_id, started_at, ended_at,"
            " hand, gesture_profile_name, difficulty, tracking_quality, pain_before, pain_after,"
            " notes_enc, demo_mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session.id,
                session.patient_id,
                session.therapist_id,
                session.consent_id,
                session.started_at,
                session.ended_at,
                session.hand.value,
                session.gesture_profile_name,
                session.difficulty,
                json.dumps(session.tracking_quality),
                session.pain_before,
                session.pain_after,
                self._cipher.encrypt(session.notes),
                int(session.demo_mode),
            ),
        )
        self._connection.commit()
        self.audit.record(
            AuditAction.SESSION_STARTED, subject_id=session.patient_id, actor_id=actor_id
        )
        return session

    def finish_session(self, session: Session, actor_id: str | None = None) -> Session:
        session.ended_at = session.ended_at or utc_now()
        self._connection.execute(
            "UPDATE session SET ended_at = ?, tracking_quality = ?, pain_before = ?,"
            " pain_after = ?, notes_enc = ? WHERE id = ?",
            (
                session.ended_at,
                json.dumps(session.tracking_quality),
                session.pain_before,
                session.pain_after,
                self._cipher.encrypt(session.notes),
                session.id,
            ),
        )
        self._connection.commit()
        self.audit.record(
            AuditAction.SESSION_FINISHED, subject_id=session.patient_id, actor_id=actor_id
        )
        return session

    def record_pain_after(self, session_id: str, pain: int) -> bool:
        """Store the post-session pain score once, only for a finished session."""
        if not 0 <= pain <= 10:
            raise ValueError("pain must be on the 0-10 scale")
        updated = self._connection.execute(
            "UPDATE session SET pain_after = ?"
            " WHERE id = ? AND ended_at IS NOT NULL AND pain_after IS NULL",
            (pain, session_id),
        )
        self._connection.commit()
        return updated.rowcount == 1

    def get_session(self, session_id: str) -> Session | None:
        row = self._connection.execute(
            "SELECT * FROM session WHERE id = ?", (session_id,)
        ).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self, patient_id: str) -> list[Session]:
        rows = self._connection.execute(
            "SELECT * FROM session WHERE patient_id = ? ORDER BY started_at", (patient_id,)
        ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def _session_from_row(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            patient_id=row["patient_id"],
            therapist_id=row["therapist_id"],
            consent_id=row["consent_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            hand=Hand(row["hand"]),
            gesture_profile_name=row["gesture_profile_name"],
            difficulty=row["difficulty"],
            tracking_quality=json.loads(row["tracking_quality"] or "{}"),
            pain_before=row["pain_before"],
            pain_after=row["pain_after"],
            notes=self._cipher.decrypt(row["notes_enc"]) or "",
            demo_mode=bool(row["demo_mode"]),
        )

    # ------------------------------------------------------- runs and metrics
    def add_exercise_run(self, run: ExerciseRun) -> ExerciseRun:
        self._connection.execute(
            "INSERT INTO exercise_run (id, session_id, kind, name, started_at, ended_at, reps,"
            " target, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.id,
                run.session_id,
                run.kind,
                run.name,
                run.started_at,
                run.ended_at,
                run.reps,
                run.target,
                json.dumps(run.payload),
            ),
        )
        self._connection.commit()
        return run

    def list_exercise_runs(self, session_id: str) -> list[ExerciseRun]:
        rows = self._connection.execute(
            "SELECT * FROM exercise_run WHERE session_id = ? ORDER BY started_at", (session_id,)
        ).fetchall()
        return [
            ExerciseRun(
                id=row["id"],
                session_id=row["session_id"],
                kind=row["kind"],
                name=row["name"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                reps=row["reps"],
                target=row["target"],
                payload=json.loads(row["payload"] or "{}"),
            )
            for row in rows
        ]

    def add_metric(self, summary: MetricSummary) -> MetricSummary:
        self._connection.execute(
            "INSERT INTO metric_summary (id, session_id, exercise_run_id, metric, payload)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                summary.id,
                summary.session_id,
                summary.exercise_run_id,
                summary.metric,
                json.dumps(summary.payload),
            ),
        )
        self._connection.commit()
        return summary

    def list_metrics(self, session_id: str) -> list[MetricSummary]:
        rows = self._connection.execute(
            "SELECT * FROM metric_summary WHERE session_id = ?", (session_id,)
        ).fetchall()
        return [
            MetricSummary(
                id=row["id"],
                session_id=row["session_id"],
                exercise_run_id=row["exercise_run_id"],
                metric=row["metric"],
                payload=json.loads(row["payload"] or "{}"),
            )
            for row in rows
        ]

    def add_samples(self, session_id: str, payloads: list[dict]) -> int:
        """Bulk-insert derived-feature samples. Never images."""
        rows = [
            (new_id(), session_id, payload.get("at", 0.0), json.dumps(payload))
            for payload in payloads
        ]
        self._connection.executemany(
            "INSERT INTO sample (id, session_id, at, payload) VALUES (?, ?, ?, ?)", rows
        )
        self._connection.commit()
        return len(rows)

    def list_samples(self, session_id: str) -> list[Sample]:
        rows = self._connection.execute(
            "SELECT * FROM sample WHERE session_id = ? ORDER BY at", (session_id,)
        ).fetchall()
        return [
            Sample(
                id=row["id"],
                session_id=row["session_id"],
                at=row["at"],
                payload=json.loads(row["payload"] or "{}"),
            )
            for row in rows
        ]

    # --------------------------------------------------------- gesture models
    def save_gesture_model(
        self,
        patient_id: str,
        model: dict[str, Any],
        bindings: dict[str, str],
        actor_id: str | None = None,
    ) -> None:
        """Store the patient's learned gestures (pose features only, never images)."""
        self._connection.execute(
            "INSERT INTO gesture_model (patient_id, model, bindings, updated_at)"
            " VALUES (?, ?, ?, ?) ON CONFLICT(patient_id) DO UPDATE SET"
            " model = excluded.model, bindings = excluded.bindings,"
            " updated_at = excluded.updated_at",
            (patient_id, json.dumps(model), json.dumps(bindings), utc_now()),
        )
        self._connection.commit()
        self.audit.record(AuditAction.GESTURES_SAVED, subject_id=patient_id, actor_id=actor_id)

    def get_gesture_model(self, patient_id: str) -> tuple[dict[str, Any], dict[str, str]] | None:
        row = self._connection.execute(
            "SELECT model, bindings FROM gesture_model WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        return (json.loads(row["model"]), json.loads(row["bindings"])) if row else None

    # ------------------------------------------------------------ user tests
    def add_participant(self, code: str, consent: bool, notes: str = "") -> None:
        """Register an anonymous test participant. The code must not identify the person."""
        self._connection.execute(
            "INSERT INTO usability_participant (code, created_at, consent, notes)"
            " VALUES (?, ?, ?, ?) ON CONFLICT(code) DO UPDATE SET consent = excluded.consent,"
            " notes = excluded.notes",
            (code, utc_now(), int(consent), notes),
        )
        self._connection.commit()

    def list_participants(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM usability_participant ORDER BY code"
        ).fetchall()
        return [dict(row) for row in rows]

    def add_usability_result(
        self, code: str, kind: str, condition: str, payload: dict[str, Any]
    ) -> str:
        if not self._connection.execute(
            "SELECT 1 FROM usability_participant WHERE code = ?", (code,)
        ).fetchone():
            self.add_participant(code, consent=False)
        result_id = new_id()
        self._connection.execute(
            "INSERT INTO usability_result (id, code, kind, condition, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (result_id, code, kind, condition, json.dumps(payload), utc_now()),
        )
        self._connection.commit()
        return result_id

    def list_usability_results(self, code: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM usability_result"
        args: tuple = ()
        if code is not None:
            query += " WHERE code = ?"
            args = (code,)
        rows = self._connection.execute(query + " ORDER BY created_at", args).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def delete_participant(self, code: str) -> None:
        self._connection.execute("DELETE FROM usability_participant WHERE code = ?", (code,))
        self._connection.commit()

    # ------------------------------------------------------ custom exercises
    def add_custom_exercise(
        self, patient_id: str, spec: dict[str, Any], actor_id: str | None = None
    ) -> str:
        """Store a validated exercise spec for one patient; returns its id."""
        exercise_id = new_id()
        self._connection.execute(
            "INSERT INTO custom_exercise (id, patient_id, therapist_id, spec, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (exercise_id, patient_id, actor_id, json.dumps(spec), utc_now()),
        )
        self._connection.commit()
        self.audit.record(AuditAction.EXERCISE_SAVED, subject_id=patient_id, actor_id=actor_id)
        return exercise_id

    def list_custom_exercises(self, patient_id: str) -> list[tuple[str, dict[str, Any]]]:
        """Active exercises of a patient as ``(id, spec)``, oldest first."""
        rows = self._connection.execute(
            "SELECT id, spec FROM custom_exercise WHERE patient_id = ? AND archived_at IS NULL"
            " ORDER BY created_at",
            (patient_id,),
        ).fetchall()
        return [(row["id"], json.loads(row["spec"])) for row in rows]

    def archive_custom_exercise(
        self, patient_id: str, exercise_id: str, actor_id: str | None = None
    ) -> bool:
        """Hide an exercise from new sessions; past sessions keep their copy of the spec."""
        updated = self._connection.execute(
            "UPDATE custom_exercise SET archived_at = ?"
            " WHERE id = ? AND patient_id = ? AND archived_at IS NULL",
            (utc_now(), exercise_id, patient_id),
        )
        self._connection.commit()
        if updated.rowcount == 1:
            self.audit.record(
                AuditAction.EXERCISE_ARCHIVED, subject_id=patient_id, actor_id=actor_id
            )
        return updated.rowcount == 1

    # ------------------------------------------------------------- documents
    def _document_path(self, document_id: str) -> Path:
        return self.documents_dir / f"{document_id}.bin"

    def add_document(
        self,
        document: PatientDocument,
        content: bytes | None = None,
        actor_id: str | None = None,
    ) -> PatientDocument:
        """Store a document. The file is encrypted and named by UUID only."""
        if content is not None:
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            self._document_path(document.id).write_bytes(self._cipher.encrypt_bytes(content))
        try:
            self._connection.execute(
                "INSERT INTO patient_document (id, patient_id, therapist_id, category,"
                " title_enc, issued_on, body_enc, filename_enc, media_type, size, sha256,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document.id,
                    document.patient_id,
                    document.therapist_id,
                    document.category,
                    self._cipher.encrypt(document.title),
                    _to_iso(document.issued_on),
                    self._cipher.encrypt(document.body),
                    self._cipher.encrypt(document.filename),
                    document.media_type or None,
                    document.size,
                    document.sha256 or None,
                    document.created_at,
                ),
            )
            self._connection.commit()
        except Exception:
            self._document_path(document.id).unlink(missing_ok=True)
            raise
        self.audit.record(
            AuditAction.DOCUMENT_ADDED, subject_id=document.patient_id, actor_id=actor_id
        )
        return document

    def _document_from_row(self, row: sqlite3.Row) -> PatientDocument:
        return PatientDocument(
            id=row["id"],
            patient_id=row["patient_id"],
            therapist_id=row["therapist_id"],
            category=row["category"],
            title=self._cipher.decrypt(row["title_enc"]) or "",
            issued_on=_from_iso(row["issued_on"]),
            body=self._cipher.decrypt(row["body_enc"]) or "",
            filename=self._cipher.decrypt(row["filename_enc"]) or "",
            media_type=row["media_type"] or "",
            size=row["size"] or 0,
            sha256=row["sha256"] or "",
            created_at=row["created_at"],
        )

    def list_documents(self, patient_id: str) -> list[PatientDocument]:
        """Documents of a patient, newest first. File contents are not loaded."""
        rows = self._connection.execute(
            "SELECT * FROM patient_document WHERE patient_id = ? ORDER BY created_at DESC",
            (patient_id,),
        ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def get_document(self, document_id: str) -> PatientDocument | None:
        row = self._connection.execute(
            "SELECT * FROM patient_document WHERE id = ?", (document_id,)
        ).fetchone()
        return self._document_from_row(row) if row else None

    def read_document_file(self, document_id: str, actor_id: str | None = None) -> bytes:
        """Decrypt an attached file, checking its integrity, and audit the access."""
        document = self.get_document(document_id)
        if document is None or not document.has_file:
            raise StorageError("unknown document")
        content = self._cipher.decrypt_bytes(self._document_path(document_id).read_bytes())
        if document.sha256 and hashlib.sha256(content).hexdigest() != document.sha256:
            raise StorageError("document integrity check failed")
        self.audit.record(
            AuditAction.DOCUMENT_VIEWED, subject_id=document.patient_id, actor_id=actor_id
        )
        return content

    def delete_document(self, document_id: str, actor_id: str | None = None) -> bool:
        document = self.get_document(document_id)
        if document is None:
            return False
        self._connection.execute("DELETE FROM patient_document WHERE id = ?", (document_id,))
        self._connection.commit()
        self._document_path(document_id).unlink(missing_ok=True)
        self.audit.record(
            AuditAction.DOCUMENT_DELETED, subject_id=document.patient_id, actor_id=actor_id
        )
        return True

    def _purge_document_files(self, patient_id: str) -> None:
        rows = self._connection.execute(
            "SELECT id FROM patient_document WHERE patient_id = ?", (patient_id,)
        ).fetchall()
        for row in rows:
            self._document_path(row["id"]).unlink(missing_ok=True)

    # ------------------------------------------------- LGPD data subject rights
    def export_patient(self, patient_id: str, actor_id: str | None = None) -> dict[str, Any]:
        """Full export of one patient's data, decrypted, for the LGPD access right."""
        patient = self.get_patient(patient_id, audit=False)
        if patient is None:
            raise StorageError("unknown patient")
        sessions = self.list_sessions(patient_id)
        export = {
            "schema_version": SCHEMA_VERSION,
            "exported_at": utc_now(),
            "patient": {
                "id": patient.id,
                "name": patient.name,
                "cpf": patient.cpf,
                "birth_date": _to_iso(patient.birth_date),
                "sex": patient.sex.value,
                "phone": patient.phone,
                "address": patient.address,
                "dominant_hand": patient.dominant_hand.value,
                "affected_hand": patient.affected_hand.value,
                "diagnosis_cid": patient.diagnosis_cid,
                "injury_date": _to_iso(patient.injury_date),
                "referring_professional": patient.referring_professional,
                "contraindications": patient.contraindications,
                "notes": patient.notes,
                "gesture_profile": patient.gesture_profile,
                "calibration": patient.calibration,
            },
            "consents": [
                {
                    "version": row["version"],
                    "accepted_at": row["accepted_at"],
                    "document_hash": row["document_hash"],
                    "revoked_at": row["revoked_at"],
                }
                for row in self._connection.execute(
                    "SELECT * FROM consent WHERE patient_id = ? ORDER BY accepted_at", (patient_id,)
                ).fetchall()
            ],
            "sessions": [
                {
                    "id": session.id,
                    "started_at": session.started_at,
                    "ended_at": session.ended_at,
                    "hand": session.hand.value,
                    "gesture_profile": session.gesture_profile_name,
                    "difficulty": session.difficulty,
                    "tracking_quality": session.tracking_quality,
                    "pain_before": session.pain_before,
                    "pain_after": session.pain_after,
                    "notes": session.notes,
                    "exercise_runs": [run.__dict__ for run in self.list_exercise_runs(session.id)],
                    "metrics": [metric.__dict__ for metric in self.list_metrics(session.id)],
                    "samples": [sample.payload for sample in self.list_samples(session.id)],
                }
                for session in sessions
            ],
            "custom_exercises": [
                {"id": exercise_id, "spec": spec}
                for exercise_id, spec in self.list_custom_exercises(patient_id)
            ],
            "gesture_model": self.get_gesture_model(patient_id),
            "documents": [
                {
                    "id": document.id,
                    "category": document.category,
                    "title": document.title,
                    "issued_on": _to_iso(document.issued_on),
                    "body": document.body,
                    "filename": document.filename,
                    "media_type": document.media_type,
                    "sha256": document.sha256,
                    "created_at": document.created_at,
                    "content_base64": (
                        base64.b64encode(
                            self._cipher.decrypt_bytes(
                                self._document_path(document.id).read_bytes()
                            )
                        ).decode("ascii")
                        if document.has_file and self._document_path(document.id).is_file()
                        else None
                    ),
                }
                for document in self.list_documents(patient_id)
            ],
        }
        self.audit.record(AuditAction.PATIENT_EXPORTED, subject_id=patient_id, actor_id=actor_id)
        return export

    def export_patient_file(
        self, patient_id: str, directory: Path, actor_id: str | None = None
    ) -> Path:
        """Write the export as JSON. **The filename uses the UUID, never the CPF.**"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.export_patient(patient_id, actor_id=actor_id)
        path = directory / f"patient_{patient_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def anonymize_patient(self, patient_id: str, actor_id: str | None = None) -> None:
        """Strip every identifier but keep the measurements for research.

        This is the option to prefer over deletion when the participant withdraws but
        agreed that anonymous data may be kept. Attached documents and written reports
        identify the person, so they are erased rather than kept.
        """
        self._purge_document_files(patient_id)
        self._connection.execute("DELETE FROM patient_document WHERE patient_id = ?", (patient_id,))
        self._connection.execute(
            "UPDATE patient SET name = ?, cpf_enc = NULL, cpf_hash = NULL, phone_enc = NULL,"
            " address_enc = NULL, contraindications_enc = NULL, notes_enc = NULL,"
            " referring_professional = '', birth_date = NULL, pin_hash = NULL,"
            " anonymized_at = ?, updated_at = ? WHERE id = ?",
            (f"Anônimo {patient_id[:8]}", utc_now(), utc_now(), patient_id),
        )
        self._connection.commit()
        self.audit.record(AuditAction.PATIENT_ANONYMIZED, subject_id=patient_id, actor_id=actor_id)

    def delete_patient(self, patient_id: str, actor_id: str | None = None) -> None:
        """Erase the patient and everything cascading from them, including document files."""
        self._purge_document_files(patient_id)
        self._connection.execute("DELETE FROM patient WHERE id = ?", (patient_id,))
        self._connection.commit()
        self.audit.record(AuditAction.PATIENT_DELETED, subject_id=patient_id, actor_id=actor_id)
