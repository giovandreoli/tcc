"""Streamlit therapist panel.

Run with ``streamlit run spectra/therapist_panel/app.py`` or via ``scripts/painel.bat``.

The panel deliberately holds no business logic: every operation goes through
:class:`spectra.service.SpectraService`, which is unit tested. What lives here is widgets
and layout only.
"""

from __future__ import annotations

from spectra.therapist_panel.app import main

__all__ = ["main"]
