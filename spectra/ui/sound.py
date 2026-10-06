"""Audio feedback.

``winsound`` only exists on Windows, so it is loaded defensively and every call is a
no-op elsewhere. This keeps the test suite silent and importable on Linux/WSL.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - platform dependent
    import winsound
except ImportError:  # pragma: no cover - non-Windows
    winsound = None


class SoundPlayer:
    """Beeps for success and error feedback; silent when unavailable or disabled."""

    SUCCESS = (880, 120)
    ERROR = (440, 160)

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @property
    def available(self) -> bool:
        return winsound is not None

    def _beep(self, frequency: int, duration_ms: int) -> bool:
        if not self.enabled or winsound is None:
            return False
        try:  # pragma: no cover - platform dependent
            winsound.Beep(frequency, duration_ms)
        except RuntimeError:
            logger.debug("winsound.Beep failed", exc_info=True)
            return False
        return True

    def success(self) -> bool:
        return self._beep(*self.SUCCESS)

    def error(self) -> bool:
        return self._beep(*self.ERROR)
