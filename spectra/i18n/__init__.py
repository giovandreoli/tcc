"""Locale catalog loader.

Code is written in English; every user-visible string lives in a JSON catalog under
``spectra/locales``. Keys are dotted paths (``paint.saved``) and values may contain
``str.format`` placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
DEFAULT_LOCALE = "pt_BR"


class Catalog:
    """An immutable, flattened set of translations for one locale."""

    def __init__(self, locale: str, entries: dict[str, str]) -> None:
        self.locale = locale
        self._entries = entries

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def t(self, key: str, **kwargs: object) -> str:
        """Translate ``key``; unknown keys are returned verbatim so bugs stay visible."""
        template = self._entries.get(key)
        if template is None:
            return key
        if not kwargs:
            return template
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template

    def keys(self) -> list[str]:
        return sorted(self._entries)


def _flatten(node: object, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}{key}." if prefix or key else key))
    else:
        flat[prefix.rstrip(".")] = str(node)
    return flat


def load_catalog(locale: str = DEFAULT_LOCALE, locales_dir: Path | None = None) -> Catalog:
    """Load ``<locale>.json``, falling back to :data:`DEFAULT_LOCALE`."""
    directory = locales_dir or LOCALES_DIR
    path = directory / f"{locale}.json"
    if not path.is_file():
        path = directory / f"{DEFAULT_LOCALE}.json"
        locale = DEFAULT_LOCALE
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Catalog(locale, _flatten(raw))


_active: Catalog | None = None


def set_locale(locale: str) -> Catalog:
    """Activate ``locale`` for the process-wide :func:`t` helper."""
    global _active
    _active = load_catalog(locale)
    return _active


def active_catalog() -> Catalog:
    if _active is None:
        return set_locale(DEFAULT_LOCALE)
    return _active


def t(key: str, **kwargs: object) -> str:
    """Translate ``key`` using the active catalog."""
    return active_catalog().t(key, **kwargs)


__all__ = ["DEFAULT_LOCALE", "Catalog", "active_catalog", "load_catalog", "set_locale", "t"]
