"""Persistence, encryption and the audit trail."""

from spectra.storage.audit import AuditLog
from spectra.storage.backup import BackupError, BackupResult, create_backup, restore_backup
from spectra.storage.db import (
    AccountLockedError,
    ConsentRequiredError,
    Database,
    DuplicateCPFError,
    StorageError,
)
from spectra.storage.models import (
    AuditAction,
    Consent,
    ExerciseRun,
    Hand,
    MetricSummary,
    Patient,
    Sample,
    Session,
    Sex,
    Therapist,
)
from spectra.storage.security import (
    Cipher,
    InvalidCPFError,
    KeyStore,
    hash_secret,
    is_valid_cpf,
    mask_cpf,
    verify_secret,
)

__all__ = [
    "AccountLockedError",
    "AuditAction",
    "AuditLog",
    "BackupError",
    "BackupResult",
    "Cipher",
    "Consent",
    "ConsentRequiredError",
    "Database",
    "DuplicateCPFError",
    "ExerciseRun",
    "Hand",
    "InvalidCPFError",
    "KeyStore",
    "MetricSummary",
    "Patient",
    "Sample",
    "Session",
    "Sex",
    "StorageError",
    "Therapist",
    "create_backup",
    "hash_secret",
    "is_valid_cpf",
    "mask_cpf",
    "restore_backup",
    "verify_secret",
]
