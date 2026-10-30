"""PDF rendering of a written clinical report (laudo) stored in the patient's record."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from spectra.i18n import t
from spectra.storage.models import PatientDocument


def written_report_pdf(
    document: PatientDocument, patient_name: str = "", therapist_name: str = ""
) -> bytes:
    styles = getSampleStyleSheet()
    styles["BodyText"].textColor = colors.HexColor("#253045")

    def paragraph(text: str, style: str = "BodyText") -> Paragraph:
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])

    issued = (
        document.issued_on.strftime("%d/%m/%Y")
        if document.issued_on
        else datetime.fromtimestamp(document.created_at).strftime("%d/%m/%Y")
    )
    story = [
        paragraph(document.title, "Title"),
        paragraph(f"{t('document.field.category')}: {t(f'document.category.{document.category}')}"),
        paragraph(f"{t('report.field.patient')}: {patient_name or '—'}"),
        paragraph(f"{t('document.field.issued_on')}: {issued}"),
        paragraph(f"{t('document.field.author')}: {therapist_name or '—'}"),
        Spacer(1, 0.4 * cm),
    ]
    story.extend(paragraph(block) for block in document.body.split("\n\n") if block.strip())
    story.extend(
        [
            Spacer(1, 1.2 * cm),
            paragraph("_" * 40),
            paragraph(therapist_name or t("document.field.signature")),
            Spacer(1, 0.6 * cm),
            paragraph(t("document.pdf_notice"), "Italic"),
        ]
    )
    buffer = BytesIO()
    SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=document.title,
        author="SPECTRA",
    ).build(story)
    return buffer.getvalue()
