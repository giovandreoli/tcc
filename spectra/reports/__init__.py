"""Generate PDF/TXT pairs and explicitly export only those documents."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from spectra.reports import pdf, txt
from spectra.reports.builder import SessionReport
from spectra.storage.db import Database
from spectra.storage.models import AuditAction


@dataclass(frozen=True)
class GeneratedReport:
    pdf_path: Path
    txt_path: Path


def generate(report: SessionReport, directory: Path) -> GeneratedReport:
    return GeneratedReport(pdf.write(report, directory), txt.write(report, directory))


def export_to_share(
    generated: GeneratedReport,
    directory: Path,
    *,
    db: Database | None = None,
    patient_id: str | None = None,
    actor_id: str | None = None,
) -> list[Path]:
    """Copy only rendered reports; the live database never enters the share."""
    sources = [Path(generated.pdf_path), Path(generated.txt_path)]
    if [source.suffix.lower() for source in sources] != [".pdf", ".txt"]:
        raise ValueError("Only PDF and TXT reports may be exported")
    if not all(source.is_file() for source in sources):
        raise FileNotFoundError("Report files are missing")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sources:
        target = destination / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    if db is not None:
        db.audit.record(AuditAction.REPORT_EXPORTED, subject_id=patient_id, actor_id=actor_id)
    return copied


__all__ = ["GeneratedReport", "export_to_share", "generate"]
