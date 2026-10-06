"""Password-protected backup of the database and the reports.

The live SQLite file stays on local disk; this module produces an encrypted *copy* that
is safe to drop into the shared folder. The password is turned into a key with scrypt and
the payload is sealed with Fernet, so a leaked backup file is not a leaked patient record.

Python's :mod:`zipfile` can only *read* encrypted archives, and its legacy ZipCrypto is
broken anyway, which is why the archive is zipped first and encrypted afterwards.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

MAGIC = b"SPECTRA-BACKUP-v1\n"
SALT_BYTES = 16
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1


class BackupError(Exception):
    """Raised when a backup cannot be written or read."""


def _derive_key(password: str, salt: bytes) -> bytes:
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
        maxmem=128 * SCRYPT_N * SCRYPT_R * 2,
    )
    return base64.urlsafe_b64encode(derived)


@dataclass(frozen=True)
class BackupResult:
    """Where the backup landed and what went into it."""

    path: Path
    entries: tuple[str, ...]
    size_bytes: int


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_backup(
    destination: Path,
    password: str,
    db_path: Path | None = None,
    extra_dirs: tuple[Path, ...] = (),
) -> BackupResult:
    """Zip the database and the given directories, then encrypt with ``password``."""
    if not password:
        raise BackupError("a backup password is required")

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    entries: list[str] = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        if db_path is not None and Path(db_path).is_file():
            archive.write(db_path, arcname=f"db/{Path(db_path).name}")
            entries.append(f"db/{Path(db_path).name}")
        for directory in extra_dirs:
            directory = Path(directory)
            if not directory.is_dir():
                continue
            for item in sorted(directory.rglob("*")):
                if item.is_file():
                    name = f"{directory.name}/{item.relative_to(directory)}"
                    archive.write(item, arcname=name)
                    entries.append(name)
        archive.writestr(
            "manifest.json",
            json.dumps({"created_at": _timestamp(), "entries": entries}, indent=2),
        )

    salt = hashlib.sha256(_timestamp().encode()).digest()[:SALT_BYTES]
    token = Fernet(_derive_key(password, salt)).encrypt(buffer.getvalue())
    path = destination / f"spectra_backup_{_timestamp()}.spectra"
    path.write_bytes(MAGIC + salt + token)
    return BackupResult(path=path, entries=tuple(entries), size_bytes=path.stat().st_size)


def restore_backup(backup_path: Path, password: str, destination: Path) -> list[str]:
    """Decrypt and unpack a backup; returns the names of the restored entries."""
    raw = Path(backup_path).read_bytes()
    if not raw.startswith(MAGIC):
        raise BackupError("not a SPECTRA backup file")
    body = raw[len(MAGIC) :]
    salt, token = body[:SALT_BYTES], body[SALT_BYTES:]
    try:
        payload = Fernet(_derive_key(password, salt)).decrypt(token)
    except InvalidToken as exc:
        raise BackupError("wrong password or corrupted backup") from exc

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(destination)
        return archive.namelist()
