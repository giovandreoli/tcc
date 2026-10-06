from __future__ import annotations

import json
from datetime import date

import pytest

from spectra.i18n import load_catalog
from spectra.reports import export_to_share, generate
from spectra.reports import pdf as pdf_renderer
from spectra.reports import txt as txt_renderer
from spectra.reports.builder import (
    ComparisonRow,
    MetricRow,
    SessionReport,
    build_session_report,
)
from spectra.reports.charts import all_charts, rom_chart
from spectra.storage.db import Database
from spectra.storage.models import (
    AuditAction,
    Consent,
    ExerciseRun,
    Hand,
    MetricSummary,
    Patient,
    Session,
    Sex,
)
from spectra.storage.security import Cipher, KeyStore

VALID_CPF = "529.982.247-25"
PNG_MAGIC = b"\x89PNG"


@pytest.fixture
def db(tmp_path) -> Database:
    cipher = Cipher.from_key_store(KeyStore(tmp_path / "secrets" / "master.key"))
    database = Database(tmp_path / "spectra.db", cipher)
    yield database
    database.close()


def seeded_session(
    db: Database, started_at: float = 1_700_000_000.0, score: float = 70.0
) -> Session:
    """One complete session with exercises and every metric family."""
    patient = db.list_patients()
    if patient:
        patient = patient[0]
    else:
        patient = db.create_patient(
            Patient(
                name="Maria da Silva",
                cpf=VALID_CPF,
                birth_date=date(1990, 5, 20),
                sex=Sex.FEMALE,
                dominant_hand=Hand.RIGHT,
                affected_hand=Hand.LEFT,
                diagnosis_cid="S62.5",
                notes="x",
            )
        )
        db.record_consent(Consent(patient_id=patient.id, version="1.0", document_hash="a" * 64))

    session = db.start_session(
        Session(patient_id=patient.id, started_at=started_at, hand=Hand.LEFT, difficulty="easy")
    )
    run = db.add_exercise_run(
        ExerciseRun(
            session_id=session.id,
            kind="physio",
            name="open_close",
            started_at=started_at,
            ended_at=started_at + 60,
            reps=8,
            target=8,
        )
    )
    db.add_metric(
        MetricSummary(
            session_id=session.id,
            exercise_run_id=run.id,
            metric="rom",
            payload={
                "fingers": [
                    {"finger": "index", "span": 70.0},
                    {"finger": "middle", "span": 65.0},
                ]
            },
        )
    )
    db.add_metric(
        MetricSummary(
            session_id=session.id,
            metric="wrist",
            payload={"estimate": True, "rotation_span": 42.0},
        )
    )
    db.add_metric(
        MetricSummary(
            session_id=session.id,
            metric="tremor",
            payload={"band_measurable": True, "band_power_ratio": 0.12, "mean_jerk": 3.4},
        )
    )
    db.add_metric(
        MetricSummary(
            session_id=session.id,
            metric="fatigue",
            payload={"amplitude_decline": 0.18, "pace_decline": 0.1},
        )
    )
    db.add_metric(
        MetricSummary(
            session_id=session.id,
            metric="accuracy",
            payload={"best_score": score, "mean_deviation": 0.021, "mean_completion": 0.88},
        )
    )
    session.ended_at = started_at + 600
    session.tracking_quality = {
        "frames_total": 1000,
        "frames_with_hand": 950,
        "mean_confidence": 0.9,
    }
    session.pain_before = 6
    session.pain_after = 3
    session.notes = "Boa adesão; orientar pausas."
    db.finish_session(session)
    return session


class TestBuilder:
    def test_the_report_carries_the_session_and_patient(self, db):
        session = seeded_session(db)
        report = build_session_report(db, session.id)
        assert report.session_id == session.id
        assert report.patient.display_name == "Maria da Silva"
        assert report.hand == "Left"
        assert report.duration_seconds == 600

    def test_the_cpf_is_always_masked(self, db):
        report = build_session_report(db, seeded_session(db).id)
        assert report.patient.masked_cpf == "***.982.247-**"
        assert "52998224725" not in str(report)

    def test_demo_mode_masks_the_name_and_the_notes(self, db):
        report = build_session_report(db, seeded_session(db).id, demo_mode=True)
        assert report.patient.display_name == "M.D.S."
        assert report.patient.masked_cpf == "***.***.***-**"
        assert report.therapist_notes == ""
        assert report.patient.demo_mode is True

    def test_metric_rows_cover_every_family(self, db):
        report = build_session_report(db, seeded_session(db).id)
        keys = {row.label_key for row in report.metrics}
        assert "report.metric.rom_finger" in keys
        assert "report.metric.wrist_rotation" in keys
        assert "report.metric.tremor_band" in keys
        assert "report.metric.amplitude_decline" in keys
        assert "report.metric.score" in keys

    def test_angular_and_wrist_rows_are_flagged_as_estimates(self, db):
        report = build_session_report(db, seeded_session(db).id)
        wrist = next(r for r in report.metrics if r.label_key == "report.metric.wrist_rotation")
        rom = next(r for r in report.metrics if r.label_key == "report.metric.rom_finger")
        assert wrist.estimate is True
        assert rom.estimate is True
        assert report.has_estimates is True

    def test_an_unmeasurable_tremor_shows_a_dash(self, db):
        session = seeded_session(db)
        db.add_metric(
            MetricSummary(
                session_id=session.id, metric="tremor", payload={"band_measurable": False}
            )
        )
        report = build_session_report(db, session.id)
        values = [r.value for r in report.metrics if r.label_key == "report.metric.tremor_band"]
        assert "—" in values

    def test_the_first_session_has_no_previous_value(self, db):
        report = build_session_report(db, seeded_session(db).id)
        assert all(row.previous is None for row in report.comparison)
        assert all(row.trend_key == "report.trend.new" for row in report.comparison)

    def test_the_second_session_is_compared_with_the_first(self, db):
        seeded_session(db, started_at=1_700_000_000.0, score=60.0)
        second = seeded_session(db, started_at=1_700_100_000.0, score=80.0)
        report = build_session_report(db, second.id)
        row = next(r for r in report.comparison if r.label_key == "report.compare.score")
        assert row.previous == 60.0
        assert row.delta == 20.0
        assert row.trend_key == "report.trend.up"
        assert report.session_number == 2
        assert report.total_sessions == 2

    def test_a_trend_series_spans_the_sessions(self, db):
        seeded_session(db, started_at=1_700_000_000.0, score=60.0)
        third = seeded_session(db, started_at=1_700_200_000.0, score=90.0)
        report = build_session_report(db, third.id)
        assert len(report.trends["report.compare.score"]) == 2

    def test_a_poorly_tracked_session_is_flagged(self, db):
        session = seeded_session(db)
        session.tracking_quality = {
            "frames_total": 1000,
            "frames_with_hand": 100,
            "mean_confidence": 0.9,
        }
        db.finish_session(session)
        report = build_session_report(db, session.id)
        assert report.reliable is False
        assert "metrics.quality.low_presence" in report.warning_keys

    def test_a_well_tracked_session_has_no_warning(self, db):
        report = build_session_report(db, seeded_session(db).id)
        assert report.reliable is True
        assert report.warning_keys == ()

    def test_generating_a_report_is_audited(self, db):
        build_session_report(db, seeded_session(db).id)
        assert db.audit.count([AuditAction.REPORT_GENERATED]) == 1

    def test_an_unknown_session_is_refused(self, db):
        with pytest.raises(ValueError):
            build_session_report(db, "nope")

    def test_the_filename_stem_uses_the_uuid(self, db):
        session = seeded_session(db)
        report = build_session_report(db, session.id)
        assert report.filename_stem == f"sessao_{session.id}"


class TestComparisonRow:
    def test_a_stable_value(self):
        assert ComparisonRow("k", 10.0, 10.0).trend_key == "report.trend.stable"

    def test_a_falling_value(self):
        row = ComparisonRow("k", 5.0, 10.0)
        assert row.delta == -5.0
        assert row.trend_key == "report.trend.down"


class TestTxtRendering:
    def test_every_section_is_present(self, db):
        text = txt_renderer.render(build_session_report(db, seeded_session(db).id))
        for key in (
            "report.section.identification",
            "report.section.session",
            "report.section.exercises",
            "report.section.metrics",
            "report.section.notes",
        ):
            assert load_catalog("pt_BR").t(key).upper() in text

    def test_the_disclaimer_is_always_included(self, db):
        text = txt_renderer.render(build_session_report(db, seeded_session(db).id))
        assert "não é um dispositivo médico certificado" in text

    def test_the_cpf_is_masked_in_the_output(self, db):
        text = txt_renderer.render(build_session_report(db, seeded_session(db).id))
        assert "***.982.247-**" in text
        assert "52998224725" not in text

    def test_estimates_are_labelled(self, db):
        text = txt_renderer.render(build_session_report(db, seeded_session(db).id))
        assert "estimativa" in text

    def test_demo_mode_hides_the_name(self, db):
        text = txt_renderer.render(build_session_report(db, seeded_session(db).id, demo_mode=True))
        assert "Maria da Silva" not in text
        assert "M.D.S." in text

    def test_an_unreliable_session_prints_a_warning(self, db):
        session = seeded_session(db)
        session.tracking_quality = {
            "frames_total": 100,
            "frames_with_hand": 5,
            "mean_confidence": 0.9,
        }
        db.finish_session(session)
        text = txt_renderer.render(build_session_report(db, session.id))
        assert "pouco confiáveis" in text

    def test_the_file_is_written_as_utf8(self, db, tmp_path):
        report = build_session_report(db, seeded_session(db).id)
        path = txt_renderer.write(report, tmp_path / "reports")
        assert path.suffix == ".txt"
        assert "Relatório" in path.read_text(encoding="utf-8")

    def test_no_untranslated_keys_leak_into_the_output(self, db):
        text = txt_renderer.render(build_session_report(db, seeded_session(db).id))
        assert "report.field." not in text
        assert "report.metric." not in text


class TestCharts:
    def test_the_rom_chart_is_a_png(self, db):
        chart = rom_chart(build_session_report(db, seeded_session(db).id))
        assert chart is not None
        assert chart.startswith(PNG_MAGIC)

    def test_charts_are_produced_for_the_session(self, db):
        charts = all_charts(build_session_report(db, seeded_session(db).id))
        keys = {key for key, _ in charts}
        assert "report.chart.rom_title" in keys
        assert "report.chart.exercises_title" in keys

    def test_a_trend_needs_at_least_two_sessions(self, db):
        single = all_charts(build_session_report(db, seeded_session(db).id))
        seeded_session(db, started_at=1_700_300_000.0, score=95.0)
        later = db.list_sessions(db.list_patients()[0].id)[-1]
        multiple = all_charts(build_session_report(db, later.id))
        assert len(multiple) > len(single)

    def test_an_empty_report_produces_no_charts(self):
        from spectra.reports.builder import ReportPatient

        empty = SessionReport(
            patient=ReportPatient(
                "A", "***", None, "patient.sex.other", "Right", "Right", "", False
            ),
            session_id="x",
            started_at=0.0,
            duration_seconds=0.0,
            hand="Right",
            gesture_profile="standard",
            difficulty="easy",
        )
        assert all_charts(empty) == []


class TestPdfRendering:
    def test_a_pdf_file_is_produced(self, db, tmp_path):
        report = build_session_report(db, seeded_session(db).id)
        path = pdf_renderer.write(report, tmp_path / "reports")
        assert path.suffix == ".pdf"
        assert path.read_bytes().startswith(b"%PDF")
        assert path.stat().st_size > 2000

    def test_the_story_contains_every_section(self, db):
        story = pdf_renderer.build_story(build_session_report(db, seeded_session(db).id))
        assert len(story) > 10

    def test_the_document_metadata_carries_no_name(self, db, tmp_path):
        report = build_session_report(db, seeded_session(db).id)
        raw = pdf_renderer.write(report, tmp_path / "reports").read_bytes()
        assert b"Maria da Silva" not in raw or report.patient.demo_mode

    def test_demo_mode_keeps_the_name_out_of_the_pdf(self, db, tmp_path):
        report = build_session_report(db, seeded_session(db).id, demo_mode=True)
        raw = pdf_renderer.write(report, tmp_path / "reports").read_bytes()
        assert b"Maria da Silva" not in raw

    def test_the_filename_uses_the_session_uuid(self, db, tmp_path):
        session = seeded_session(db)
        path = pdf_renderer.write(build_session_report(db, session.id), tmp_path / "reports")
        assert session.id in path.name
        assert "52998224725" not in path.name


class TestGenerationAndExport:
    def test_both_formats_are_generated_together(self, db, tmp_path):
        report = build_session_report(db, seeded_session(db).id)
        generated = generate(report, tmp_path / "reports")
        assert generated.pdf_path.exists()
        assert generated.txt_path.exists()

    def test_export_copies_to_the_share_folder_and_audits(self, db, tmp_path):
        session = seeded_session(db)
        patient_id = db.list_patients()[0].id
        generated = generate(build_session_report(db, session.id), tmp_path / "reports")
        copied = export_to_share(generated, tmp_path / "share", db=db, patient_id=patient_id)
        assert len(copied) == 2
        assert all(path.exists() for path in copied)
        assert db.audit.count([AuditAction.REPORT_EXPORTED]) == 1

    def test_the_database_itself_is_never_exported(self, db, tmp_path):
        session = seeded_session(db)
        generated = generate(build_session_report(db, session.id), tmp_path / "reports")
        export_to_share(generated, tmp_path / "share")
        assert list((tmp_path / "share").glob("*.db")) == []

    def test_the_share_folder_is_created_if_missing(self, db, tmp_path):
        generated = generate(build_session_report(db, seeded_session(db).id), tmp_path / "reports")
        export_to_share(generated, tmp_path / "nested" / "share")
        assert (tmp_path / "nested" / "share").is_dir()


class TestCatalogCompleteness:
    @pytest.mark.parametrize(
        "key",
        [
            "report.title",
            "report.disclaimer",
            "report.section.metrics",
            "report.field.pain_before",
            "report.metric.rom_finger",
            "report.trend.up",
            "report.chart.rom_title",
            "patient.sex.female",
            "metrics.quality.low_presence",
            "common.estimate",
        ],
    )
    def test_required_report_keys_exist(self, key):
        assert key in load_catalog("pt_BR")

    def test_every_metric_label_resolves(self, db):
        catalog = load_catalog("pt_BR")
        report = build_session_report(db, seeded_session(db).id)
        for row in report.metrics:
            assert row.label_key in catalog

    def test_the_locale_file_is_valid_json(self):
        from spectra.i18n import LOCALES_DIR

        json.loads((LOCALES_DIR / "pt_BR.json").read_text(encoding="utf-8"))


class TestMetricRowDefaults:
    def test_a_plain_row_is_not_an_estimate(self):
        assert MetricRow("k", "1").estimate is False
        assert MetricRow("k", "1").label_args == {}
