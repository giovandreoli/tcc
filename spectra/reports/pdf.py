"""ReportLab rendering of already privacy-filtered report data."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

from spectra.i18n import Catalog, active_catalog
from spectra.reports.builder import SessionReport
from spectra.reports.charts import all_charts
from spectra.reports.txt import sections


def build_story(report: SessionReport, catalog: Catalog | None = None) -> list:
    c = catalog or active_catalog()
    styles = getSampleStyleSheet()
    styles["BodyText"].textColor = colors.HexColor("#253045")

    def paragraph(text: str, style: str = "BodyText") -> Paragraph:
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])

    story = [paragraph(c.t("report.title"), "Title")]
    if report.patient.demo_mode:
        story.append(paragraph(c.t("report.demo_mode_notice")))
    for key, lines in sections(report, c):
        story.append(paragraph(c.t(key), "Heading2"))
        story.extend(paragraph(line) for line in lines)
        story.append(Spacer(1, 0.15 * cm))
    charts = all_charts(report)
    if charts:
        story.append(paragraph(c.t("report.section.charts"), "Heading2"))
        for key, png in charts:
            story.extend(
                [
                    paragraph(c.t(key), "Heading3"),
                    Image(BytesIO(png), width=16 * cm, height=7.3 * cm),
                ]
            )
    story.extend([Spacer(1, 0.4 * cm), paragraph(c.t("report.disclaimer"))])
    return story


def write(report: SessionReport, directory: Path, catalog: Catalog | None = None) -> Path:
    path = Path(directory) / f"{report.filename_stem}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="SPECTRA",
        author="SPECTRA",
        subject=report.session_id,
        pageCompression=1,
    )
    document.build(build_story(report, catalog))
    return path
