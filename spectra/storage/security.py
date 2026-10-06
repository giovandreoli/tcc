"""Security primitives: CPF handling, encryption at rest, key storage and KDF hashing.

LGPD treats health data as sensitive personal data. The rules this module enforces:

* **CPF never appears in plaintext at rest, in a filename, in a log line or in an
  exception message.** Records are keyed by UUID. Lookup by CPF goes through a keyed
  (HMAC) hash, so the index is searchable without being reversible.
* Free-text clinical fields, phone and address are encrypted with Fernet (AES-128-CBC
  plus HMAC-SHA256).
* The key lives outside the repository: in the OS credential store via ``keyring`` when
  available, otherwise in a 0600 file inside the data directory.
* PINs and passwords are hashed with scrypt, which is memory-hard. ``hashlib.scrypt`` is
  in the standard library, so this adds no dependency.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from spectra.storage.errors import StorageError

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "spectra"
KEYRING_USERNAME = "master-key"

#: scrypt parameters. n=2**15 costs ~32 MB and ~100 ms, comfortably above the
#: 2017 OWASP floor while staying usable on the author's laptop.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16


def _maxmem(n: int, r: int) -> int:
    """OpenSSL defaults to a 32 MB cap, which these parameters exceed; raise it."""
    return 128 * n * r * 2


#: Failed attempts tolerated before an account is locked, and for how long.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300


class SecurityError(StorageError):
    """Raised when a security invariant is violated."""


# ------------------------------------------------------------------------- CPF
class InvalidCPFError(SecurityError):
    """Raised for a malformed CPF. **Never carries the offending value.**"""

    def __init__(self) -> None:
        super().__init__("invalid CPF")


def normalize_cpf(cpf: str) -> str:
    """Strip formatting, leaving 11 digits."""
    return "".join(character for character in str(cpf) if character.isdigit())


def _check_digit(digits: str, weight: int) -> str:
    total = sum(int(digit) * (weight - index) for index, digit in enumerate(digits))
    remainder = (total * 10) % 11
    return "0" if remainder == 10 else str(remainder)


def is_valid_cpf(cpf: str) -> bool:
    """Validate length, the repeated-digit case and both check digits."""
    digits = normalize_cpf(cpf)
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    first = _check_digit(digits[:9], 10)
    second = _check_digit(digits[:9] + first, 11)
    return digits[9:] == first + second


def validate_cpf(cpf: str) -> str:
    """Return the normalised CPF or raise :class:`InvalidCPFError`."""
    if not is_valid_cpf(cpf):
        raise InvalidCPFError
    return normalize_cpf(cpf)


def mask_cpf(cpf: str) -> str:
    """Mask a CPF for display: ``***.456.789-**``."""
    digits = normalize_cpf(cpf)
    if len(digits) != 11:
        return "***.***.***-**"
    return f"***.{digits[3:6]}.{digits[6:9]}-**"


def format_cpf(cpf: str) -> str:
    """Format a CPF in full. Only ever used on screen for the therapist."""
    digits = normalize_cpf(cpf)
    if len(digits) != 11:
        return digits
    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"


def cpf_lookup_hash(cpf: str, key: bytes) -> str:
    """Deterministic keyed hash of a CPF, for an index that cannot be reversed.

    Keyed rather than plain SHA-256: the CPF space is small enough (10^11) that an
    unkeyed digest could be brute-forced in minutes.
    """
    digits = normalize_cpf(cpf)
    return hmac.new(key, digits.encode("utf-8"), hashlib.sha256).hexdigest()


# ------------------------------------------------------------------- key store
@dataclass
class KeyStore:
    """Loads or creates the master key, preferring the OS credential store."""

    fallback_path: Path
    service: str = KEYRING_SERVICE
    username: str = KEYRING_USERNAME

    def load(self) -> bytes:
        key = self._from_keyring() or self._from_file()
        if key is None:
            key = self._create()
        return key

    # The keyring backend is absent on headless Linux and in CI; the file fallback
    # keeps the app working there, and docs/lgpd.md documents the weaker guarantee.
    def _from_keyring(self) -> bytes | None:
        try:
            import keyring

            stored = keyring.get_password(self.service, self.username)
        except Exception:
            logger.debug("keyring unavailable, falling back to the key file", exc_info=True)
            return None
        return stored.encode("ascii") if stored else None

    def _from_file(self) -> bytes | None:
        if not self.fallback_path.is_file():
            return None
        return self.fallback_path.read_bytes().strip() or None

    def _create(self) -> bytes:
        key = Fernet.generate_key()
        try:
            import keyring

            keyring.set_password(self.service, self.username, key.decode("ascii"))
            logger.info("master key stored in the OS credential store")
            return key
        except Exception:
            logger.warning("keyring unavailable; storing the master key in %s", self.fallback_path)
        self._write_file(key)
        return key

    def _write_file(self, key: bytes) -> None:
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path.write_bytes(key)
        try:
            os.chmod(self.fallback_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:  # pragma: no cover - Windows ACLs differ
            logger.debug("could not restrict permissions on the key file", exc_info=True)


class Cipher:
    """Encrypts and decrypts the sensitive text columns."""

    def __init__(self, key: bytes) -> None:
        self._fernet = Fernet(key)
        self._key = key

    @classmethod
    def from_key_store(cls, store: KeyStore) -> Cipher:
        return cls(store.load())

    @property
    def lookup_key(self) -> bytes:
        """A separate key for CPF hashing, derived so it cannot decrypt anything."""
        return hashlib.sha256(b"spectra-lookup:" + self._key).digest()

    def encrypt(self, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str | None) -> str | None:
        if not token:
            return None
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise SecurityError("could not decrypt a stored field") from exc

    def cpf_hash(self, cpf: str) -> str:
        return cpf_lookup_hash(cpf, self.lookup_key)


# ----------------------------------------------------------------------- KDF
def hash_secret(secret: str) -> str:
    """Hash a PIN or password with scrypt; returns a self-describing string."""
    salt = secrets.token_bytes(SALT_BYTES)
    derived = hashlib.scrypt(
        secret.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
        maxmem=_maxmem(SCRYPT_N, SCRYPT_R),
    )
    return "$".join(
        (
            "scrypt",
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )
    )


def verify_secret(secret: str, encoded: str | None) -> bool:
    """Constant-time check of ``secret`` against a stored hash."""
    if not encoded:
        return False
    try:
        algorithm, n, r, p, salt_b64, hash_b64 = encoded.split("$")
        if algorithm != "scrypt":
            return False
        derived = hashlib.scrypt(
            secret.encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(base64.b64decode(hash_b64)),
            maxmem=_maxmem(int(n), int(r)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, base64.b64decode(hash_b64))


def is_valid_pin(pin: str) -> bool:
    """A patient PIN is exactly four digits."""
    return len(pin) == 4 and pin.isdigit()


# ------------------------------------------------------------------- lockout
@dataclass(frozen=True)
class LockoutState:
    """Result of an authentication attempt against a lockable account."""

    authenticated: bool
    locked: bool
    failed_attempts: int
    locked_until: float | None = None

    @property
    def remaining_attempts(self) -> int:
        return max(0, MAX_FAILED_ATTEMPTS - self.failed_attempts)


def next_lockout(
    success: bool,
    failed_attempts: int,
    now: float,
    max_attempts: int = MAX_FAILED_ATTEMPTS,
    lockout_seconds: int = LOCKOUT_SECONDS,
) -> LockoutState:
    """Pure lockout transition, so the policy is testable without a database."""
    if success:
        return LockoutState(authenticated=True, locked=False, failed_attempts=0)
    attempts = failed_attempts + 1
    if attempts >= max_attempts:
        return LockoutState(
            authenticated=False,
            locked=True,
            failed_attempts=attempts,
            locked_until=now + lockout_seconds,
        )
    return LockoutState(authenticated=False, locked=False, failed_attempts=attempts)


def is_locked(locked_until: float | None, now: float) -> bool:
    return locked_until is not None and now < locked_until
