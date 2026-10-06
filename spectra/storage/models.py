"""Domain records.

Field-level privacy rules live here, in :data:`ENCRYPTED_FIELDS`, so the database layer
cannot forget one. Patients are identified by a UUID; the CPF is stored encrypted plus a
keyed hash for lookup, and never appears in a filename or a log line.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> float:
    return datetime.now(timezone.utc).timestamp()


class Hand(Enum):
    LEFT = "Left"
    RIGHT = "Right"


class Sex(Enum):
    """Values double as i18n keys under ``patient.sex.``."""

    FEMALE = "female"
    MALE = "male"
    OTHER = "other"
    UNDISCLOSED = "undisclosed"


#: Columns encrypted at rest. Chosen by the data-minimisation rule: anything that
#: identifies the person outside the clinic, plus free-text clinical content.
ENCRYPTED_FIELDS: tuple[str, ...] = (
    "cpf",
    "phone",
    "address",
    "contraindications",
    "notes",
)


@dataclass
class Therapist:
    """A professional account. The password hash never leaves this layer."""

    id: str = field(default_factory=new_id)
    name: str = ""
    email: str = ""
    password_hash: str | None = None
    created_at: float = field(default_factory=utc_now)
    failed_attempts: int = 0
    locked_until: float | None = None
    is_active: bool = True


@dataclass
class Patient:
    """A patient record. Only fields with a clinical or measurement purpose are kept."""

    id: str = field(default_factory=new_id)
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
    gesture_profile: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    pin_hash: str | None = None
    failed_attempts: int = 0
    locked_until: float | None = None
    created_at: float = field(default_factory=utc_now)
    updated_at: float = field(default_factory=utc_now)
    anonymized_at: float | None = None

    @property
    def is_anonymized(self) -> bool:
        return self.anonymized_at is not None

    def initials(self) -> str:
        """Pseudonym used in demo mode and in masked reports."""
        parts = [part for part in self.name.split() if part]
        if not parts:
            return "—"
        return ".".join(part[0].upper() for part in parts) + "."

    def display_name(self, demo_mode: bool = False) -> str:
        return self.initials() if demo_mode else self.name

    def age_at(self, moment: date | None = None) -> int | None:
        if self.birth_date is None:
            return None
        reference = moment or date.today()
        years = reference.year - self.birth_date.year
        if (reference.month, reference.day) < (self.birth_date.month, self.birth_date.day):
            years -= 1
        return years


@dataclass
class Consent:
    """A recorded TCLE acceptance. No session may be created without one."""

    id: str = field(default_factory=new_id)
    patient_id: str = ""
    version: str = ""
    accepted_at: float = field(default_factory=utc_now)
    #: SHA-256 of the exact consent text the patient saw, so it can be proven later.
    document_hash: str = ""
    signed_by: str = ""
    revoked_at: float | None = None

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


@dataclass
class Session:
    """One rehabilitation session."""

    id: str = field(default_factory=new_id)
    patient_id: str = ""
    therapist_id: str | None = None
    consent_id: str | None = None
    started_at: float = field(default_factory=utc_now)
    ended_at: float | None = None
    hand: Hand = Hand.RIGHT
    gesture_profile_name: str = "standard"
    difficulty: str = "easy"
    tracking_quality: dict = field(default_factory=dict)
    pain_before: int | None = None
    pain_after: int | None = None
    notes: str = ""
    demo_mode: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, (self.ended_at or self.started_at) - self.started_at)


@dataclass
class ExerciseRun:
    """One exercise or guided-drawing attempt inside a session."""

    id: str = field(default_factory=new_id)
    session_id: str = ""
    kind: str = "physio"
    name: str = ""
    started_at: float = field(default_factory=utc_now)
    ended_at: float | None = None
    reps: int = 0
    target: int = 0
    payload: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, (self.ended_at or self.started_at) - self.started_at)


@dataclass
class MetricSummary:
    """A computed metric, stored as JSON so new metrics need no migration."""

    id: str = field(default_factory=new_id)
    session_id: str = ""
    exercise_run_id: str | None = None
    metric: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class Sample:
    """One stored derived-feature frame. Never an image."""

    id: str = field(default_factory=new_id)
    session_id: str = ""
    at: float = 0.0
    payload: dict = field(default_factory=dict)


class AuditAction(Enum):
    """Auditable actions; values double as i18n keys under ``audit.action.``."""

    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    ACCOUNT_LOCKED = "account_locked"
    PATIENT_CREATED = "patient_created"
    PATIENT_UPDATED = "patient_updated"
    PATIENT_VIEWED = "patient_viewed"
    PATIENT_EXPORTED = "patient_exported"
    PATIENT_ANONYMIZED = "patient_anonymized"
    PATIENT_DELETED = "patient_deleted"
    CONSENT_RECORDED = "consent_recorded"
    CONSENT_REVOKED = "consent_revoked"
    SESSION_STARTED = "session_started"
    SESSION_FINISHED = "session_finished"
    REPORT_GENERATED = "report_generated"
    REPORT_EXPORTED = "report_exported"
    BACKUP_CREATED = "backup_created"


@dataclass
class AuditEntry:
    """One immutable audit-log line."""

    id: str = field(default_factory=new_id)
    at: float = field(default_factory=utc_now)
    actor_type: str = "therapist"
    actor_id: str | None = None
    action: AuditAction = AuditAction.PATIENT_VIEWED
    subject_type: str = "patient"
    subject_id: str | None = None
    detail: str = ""
