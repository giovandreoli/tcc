"""The informed-consent (TCLE) document and its versioning.

The text lives in a file rather than in code so that changing the wording is a visible,
reviewable diff. Every consent record stores a SHA-256 of the exact text the participant
saw, so it can later be proven which version was agreed to.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

CONSENT_DIR = Path(__file__).resolve().parent / "locales"
CURRENT_VERSION = "1.0"


@dataclass(frozen=True)
class ConsentDocument:
    """A versioned consent text plus the hash that gets stored with the acceptance."""

    version: str
    text: str

    @property
    def document_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_consent(locale: str = "pt_BR", version: str = CURRENT_VERSION) -> ConsentDocument:
    """Load the consent text for ``locale``; falls back to pt-BR."""
    path = CONSENT_DIR / f"tcle_{locale}.md"
    if not path.is_file():
        path = CONSENT_DIR / "tcle_pt_BR.md"
    return ConsentDocument(version=version, text=path.read_text(encoding="utf-8"))
