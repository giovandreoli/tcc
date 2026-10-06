"""Append-only audit log.

LGPD requires that access to, export of and deletion of personal data be traceable. The
log stores **UUIDs only** — never a name, never a CPF — so the log itself is not a second
copy of the sensitive data.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

from spectra.storage.models import AuditAction, AuditEntry, new_id, utc_now


class AuditLog:
    """Writes and reads audit entries on the caller's SQLite connection."""

    def __init__(self, connection: Callable[[], sqlite3.Connection]) -> None:
        self._get_connection = connection

    @property
    def _connection(self) -> sqlite3.Connection:
        return self._get_connection()

    def record(
        self,
        action: AuditAction,
        subject_id: str | None = None,
        subject_type: str = "patient",
        actor_id: str | None = None,
        actor_type: str = "therapist",
        detail: str = "",
    ) -> AuditEntry:
        entry = AuditEntry(
            id=new_id(),
            at=utc_now(),
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            subject_type=subject_type,
            subject_id=subject_id,
            detail=detail,
        )
        self._connection.execute(
            "INSERT INTO audit_log (id, at, actor_type, actor_id, action, subject_type,"
            " subject_id, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.id,
                entry.at,
                entry.actor_type,
                entry.actor_id,
                entry.action.value,
                entry.subject_type,
                entry.subject_id,
                entry.detail,
            ),
        )
        self._connection.commit()
        return entry

    def entries(self, subject_id: str | None = None, limit: int = 200) -> list[AuditEntry]:
        if subject_id is None:
            rows = self._connection.execute(
                "SELECT * FROM audit_log ORDER BY at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM audit_log WHERE subject_id = ? ORDER BY at DESC LIMIT ?",
                (subject_id, limit),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: sqlite3.Row) -> AuditEntry:
        return AuditEntry(
            id=row["id"],
            at=row["at"],
            actor_type=row["actor_type"],
            actor_id=row["actor_id"],
            action=AuditAction(row["action"]),
            subject_type=row["subject_type"],
            subject_id=row["subject_id"],
            detail=row["detail"] or "",
        )

    def count(self, actions: Sequence[AuditAction] | None = None) -> int:
        if not actions:
            return int(self._connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])
        placeholders = ",".join("?" for _ in actions)
        return int(
            self._connection.execute(
                f"SELECT COUNT(*) FROM audit_log WHERE action IN ({placeholders})",
                [action.value for action in actions],
            ).fetchone()[0]
        )
