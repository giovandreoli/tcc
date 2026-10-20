"""Expiring room codes and exclusive admission for local browser sessions."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass

from spectra.service import SessionPlan, SpectraService
from spectra.storage.db import AccountLockedError
from spectra.storage.models import Session, utc_now

EXERCISES = ("open_close", "finger_touch", "finger_wave")
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
ROOM_LIFETIME_SECONDS = 3600
LEASE_SECONDS = 90


class RoomError(RuntimeError):
    """Carries a locale key, never a code, PIN or patient identifier."""


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_code(code: str) -> str:
    return code.replace("-", "").replace(" ", "").upper()


@dataclass(frozen=True)
class RoomAdmission:
    session: Session
    code_hash: str
    claim: str
    duration_minutes: int
    exercises: tuple[str, ...]


@dataclass(frozen=True)
class RoomInfo:
    """A room as seen by the therapist monitor. Holds no code, claim or PIN."""

    session_id: str
    patient_id: str
    status: str
    created_at: float
    expires_at: float


class Rooms:
    def __init__(self, service: SpectraService) -> None:
        self.service = service

    def overview(self, window_seconds: float = 12 * 3600) -> list[RoomInfo]:
        """Rooms created in the recent window, newest first, with a derived status."""
        now = utc_now()
        with self.service.db.transaction() as connection:
            rows = connection.execute(
                "SELECT r.session_id, r.status, r.expires_at, r.heartbeat_at, s.patient_id"
                " FROM browser_room r JOIN session s ON s.id = r.session_id"
                " WHERE r.expires_at > ? ORDER BY r.expires_at DESC",
                (now - window_seconds + ROOM_LIFETIME_SECONDS,),
            ).fetchall()
        rooms = []
        for row in rows:
            status = row["status"]
            if status == "active" and (row["heartbeat_at"] or 0) < now - LEASE_SECONDS:
                status = "interrupted"
            elif status == "waiting" and row["expires_at"] <= now:
                status = "expired"
            rooms.append(
                RoomInfo(
                    session_id=row["session_id"],
                    patient_id=row["patient_id"],
                    status=status,
                    created_at=row["expires_at"] - ROOM_LIFETIME_SECONDS,
                    expires_at=row["expires_at"],
                )
            )
        return rooms

    def create(self, plan: SessionPlan) -> str:
        if (
            not plan.exercises
            or any(exercise not in EXERCISES for exercise in plan.exercises)
            or not 5 <= plan.duration_minutes <= 40
        ):
            raise RoomError("browser.invalid_plan")
        patient = self.service.db.get_patient(plan.patient_id, audit=False)
        if patient is None or not patient.pin_hash:
            raise RoomError("browser.pin_required")
        session = self.service.start_session(plan)
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(12))
        with self.service.db.transaction() as connection:
            connection.execute(
                "INSERT INTO browser_room"
                " (code_hash, session_id, expires_at, duration_minutes, exercises)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    digest(code),
                    session.id,
                    utc_now() + ROOM_LIFETIME_SECONDS,
                    plan.duration_minutes,
                    json.dumps(plan.exercises),
                ),
            )
        return "-".join(code[i : i + 4] for i in range(0, 12, 4))

    def admit(self, code: str, pin: str) -> RoomAdmission:
        code_hash = digest(normalize_code(code))
        with self.service.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM browser_room WHERE code_hash = ?", (code_hash,)
            ).fetchone()
        now = utc_now()
        if row is None or row["expires_at"] <= now or row["status"] == "completed":
            raise RoomError("browser.invalid_room")
        session = self.service.db.get_session(row["session_id"])
        if (
            session is None
            or session.ended_at is not None
            or not self.service.has_consent(session.patient_id)
        ):
            raise RoomError("browser.invalid_room")
        try:
            authentication = self.service.db.authenticate_patient(session.patient_id, pin)
        except AccountLockedError as exc:
            raise RoomError("browser.locked") from exc
        if not authentication.authenticated:
            raise RoomError("browser.invalid_pin")
        claim = secrets.token_urlsafe(32)
        with self.service.db.transaction() as connection:
            updated = connection.execute(
                "UPDATE browser_room SET status = 'active', claim_hash = ?, heartbeat_at = ?"
                " WHERE code_hash = ? AND expires_at > ? AND (status = 'waiting'"
                " OR (status = 'active' AND heartbeat_at < ?))",
                (digest(claim), now, code_hash, now, now - LEASE_SECONDS),
            )
            if updated.rowcount != 1:
                raise RoomError("browser.in_use")
        return RoomAdmission(
            session, code_hash, claim, row["duration_minutes"], tuple(json.loads(row["exercises"]))
        )

    def heartbeat(self, admission: RoomAdmission) -> None:
        session = self.service.db.get_session(admission.session.id)
        if (
            session is None
            or session.ended_at is not None
            or not self.service.has_consent(session.patient_id)
        ):
            raise RoomError("browser.invalid_room")
        with self.service.db.transaction() as connection:
            updated = connection.execute(
                "UPDATE browser_room SET heartbeat_at = ?"
                " WHERE code_hash = ? AND claim_hash = ? AND status = 'active'",
                (utc_now(), admission.code_hash, digest(admission.claim)),
            )
            if updated.rowcount != 1:
                raise RoomError("browser.in_use")

    def complete(self, admission: RoomAdmission) -> None:
        with self.service.db.transaction() as connection:
            updated = connection.execute(
                "UPDATE browser_room SET status = 'completed', claim_hash = NULL"
                " WHERE code_hash = ? AND claim_hash = ? AND status = 'active'",
                (admission.code_hash, digest(admission.claim)),
            )
            if updated.rowcount != 1:
                raise RoomError("browser.invalid_room")

    def release(self, admission: RoomAdmission) -> None:
        """Return an unused admission to the waiting state if it still owns the room."""
        with self.service.db.transaction() as connection:
            connection.execute(
                "UPDATE browser_room SET status = 'waiting', claim_hash = NULL,"
                " heartbeat_at = NULL WHERE code_hash = ? AND claim_hash = ?",
                (admission.code_hash, digest(admission.claim)),
            )
