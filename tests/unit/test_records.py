"""Patient documents, custom exercise storage, landmark batches and remote access."""

from __future__ import annotations

import base64
import hashlib
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from spectra.browser.frames import MAX_FRAMES_PER_BATCH, BatchError, parse_batch
from spectra.exercises import ExerciseSpec, Movement
from spectra.remote import (
    TUNNEL_URL,
    Tunnel,
    TunnelError,
    binary_name,
    download_cloudflared,
    find_cloudflared,
    is_local_request,
    patient_link,
)
from spectra.service import MAX_DOCUMENT_BYTES, DocumentForm, PatientForm, SpectraService
from spectra.storage.db import Database
from spectra.storage.errors import StorageError
from spectra.storage.models import AuditAction
from spectra.storage.security import Cipher

PDF = b"%PDF-1.4\n%fake test pdf\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture
def service(config):
    from cryptography.fernet import Fernet

    instance = SpectraService(Database(config.db_path, Cipher(Fernet.generate_key())), config)
    yield instance
    instance.close()


@pytest.fixture
def patient(service):
    return service.create_patient(PatientForm(name="Maria Teste", cpf="52998224725", pin="1234"))


# ---------------------------------------------------------------- documents
def test_written_report_is_encrypted_at_rest_and_listed(service, patient):
    therapist = service.db.create_therapist("Dra. Ana", "ana@example.com", "senha-segura")
    form = DocumentForm(title="Laudo inicial", body="Rigidez de IFP do indicador direito.")
    document = service.add_document(patient.id, form, actor_id=therapist.id)
    with service.db.transaction() as connection:
        row = connection.execute("SELECT * FROM patient_document").fetchone()
    raw = " ".join(str(value) for value in tuple(row))
    assert "Laudo inicial" not in raw and "Rigidez" not in raw
    [listed] = service.db.list_documents(patient.id)
    assert listed.title == "Laudo inicial" and listed.body.startswith("Rigidez")
    assert listed.therapist_id == therapist.id and not listed.has_file
    assert document.id == listed.id


def test_uploaded_file_is_encrypted_named_by_uuid_and_integrity_checked(service, patient):
    form = DocumentForm(
        category="medical_report",
        title="Laudo do ortopedista",
        filename="C:\\fakepath\\laudo Maria 529.982.247-25.pdf",
        content=PDF,
    )
    document = service.add_document(patient.id, form)
    files = list(service.db.documents_dir.iterdir())
    assert [path.name for path in files] == [f"{document.id}.bin"]
    assert PDF not in files[0].read_bytes()
    assert document.filename == "laudo Maria 529.982.247-25.pdf"  # Path stripped.
    assert document.media_type == "application/pdf"
    assert document.sha256 == hashlib.sha256(PDF).hexdigest()
    assert service.db.read_document_file(document.id) == PDF
    assert service.db.audit.count([AuditAction.DOCUMENT_VIEWED]) == 1
    files[0].write_bytes(service.db._cipher.encrypt_bytes(b"%PDF-tampered"))
    with pytest.raises(StorageError, match="integrity"):
        service.db.read_document_file(document.id)


@pytest.mark.parametrize(
    ("form", "problem"),
    [
        (DocumentForm(title="ab", body="x"), "document.error.title"),
        (DocumentForm(title="Laudo"), "document.error.empty"),
        (DocumentForm(title="Laudo", body="x", category="bogus"), "document.error.category"),
        (DocumentForm(title="Laudo", content=b"MZ\x90\x00 executable"), "document.error.type"),
        (
            DocumentForm(title="Laudo", content=PDF + b"0" * MAX_DOCUMENT_BYTES),
            "document.error.too_large",
        ),
        (
            DocumentForm(title="Laudo", body="x", issued_on=date.today() + timedelta(days=1)),
            "document.error.date_future",
        ),
    ],
)
def test_invalid_documents_are_rejected(service, patient, form, problem):
    with pytest.raises(ValueError) as error:
        service.add_document(patient.id, form)
    assert problem in error.value.args[0]
    assert service.db.list_documents(patient.id) == []


def test_written_report_renders_as_pdf(service, patient):
    therapist = service.db.create_therapist("Dra. Ana", "ana@example.com", "senha-segura")
    document = service.add_document(
        patient.id,
        DocumentForm(title="Laudo fisioterapêutico", body="Avaliação.\n\nConduta: <b>x</b>"),
        actor_id=therapist.id,
    )
    pdf = service.written_report_pdf(document.id, actor_id=therapist.id)
    assert pdf.startswith(b"%PDF")


def test_deleting_a_document_removes_its_file(service, patient):
    document = service.add_document(patient.id, DocumentForm(title="Exame", content=PNG))
    assert service.db.delete_document(document.id)
    assert not any(service.db.documents_dir.iterdir())
    assert not service.db.delete_document(document.id)
    assert service.db.audit.count([AuditAction.DOCUMENT_DELETED]) == 1


def test_lgpd_export_includes_documents_and_custom_exercises(service, patient):
    service.add_document(patient.id, DocumentForm(title="Exame", content=PNG, body="obs"))
    service.save_custom_exercise(
        patient.id, ExerciseSpec("", Movement.OPEN_CLOSE, title="Punho firme")
    )
    export = service.db.export_patient(patient.id)
    [document] = export["documents"]
    assert base64.b64decode(document["content_base64"]) == PNG
    assert document["body"] == "obs"
    assert export["custom_exercises"][0]["spec"]["title"] == "Punho firme"


@pytest.mark.parametrize("operation", ["anonymize_patient", "delete_patient"])
def test_anonymization_and_deletion_erase_documents(service, patient, operation):
    service.add_document(patient.id, DocumentForm(title="Exame", content=PNG))
    getattr(service.db, operation)(patient.id)
    assert service.db.list_documents(patient.id) == []
    assert not any(service.db.documents_dir.iterdir())


def test_backup_includes_encrypted_documents(service, patient, tmp_path):
    from spectra.storage.backup import restore_backup

    service.add_document(patient.id, DocumentForm(title="Exame", content=PNG))
    result = service.backup("senha-do-backup", destination=tmp_path / "out")
    restored = restore_backup(result.path, "senha-do-backup", tmp_path / "restored")
    assert any(name.startswith("documents/") and name.endswith(".bin") for name in restored)


def test_database_upgrades_an_existing_version_1_file(config):
    import sqlite3

    from cryptography.fernet import Fernet

    key = Fernet.generate_key()
    Database(config.db_path, Cipher(key)).close()
    with sqlite3.connect(config.db_path) as connection:
        connection.execute("DROP TABLE patient_document")
        connection.execute("DROP TABLE custom_exercise")
        connection.execute("UPDATE schema_info SET version = 1")
    database = Database(config.db_path, Cipher(key))
    assert database.list_documents("nobody") == []
    assert database._connection.execute("SELECT version FROM schema_info").fetchone()[0] == 3
    database.close()


# --------------------------------------------------------- custom exercises
def test_custom_exercises_are_validated_scoped_and_archived(service, patient):
    with pytest.raises(ValueError) as error:
        service.save_custom_exercise(patient.id, ExerciseSpec("", Movement.PINCH, title="x"))
    assert "exercise.error.title" in error.value.args[0]
    spec = service.save_custom_exercise(
        patient.id,
        ExerciseSpec(
            "", Movement.PINCH, title="  Pinça média  ", finger="middle", instructions=("", " a ")
        ),
    )
    assert spec.key.startswith("custom:") and spec.title == "Pinça média"
    assert spec.instructions == ("a",)
    keys = [item.key for item in service.available_exercises(patient.id)]
    assert keys == ["open_close", "finger_touch", "finger_wave", spec.key]
    assert service.resolve_exercises(patient.id, (spec.key,))[0].finger == "middle"
    assert service.archive_custom_exercise(patient.id, spec.key)
    assert not service.archive_custom_exercise(patient.id, spec.key)
    assert not service.archive_custom_exercise(patient.id, "open_close")
    assert service.custom_exercises(patient.id) == []
    with pytest.raises(ValueError):
        service.resolve_exercises(patient.id, (spec.key,))


# -------------------------------------------------------------- landmarks
def _row(at=1.0, label="Right", score=0.9, coordinates=None):
    return [at, label, score, coordinates if coordinates is not None else [0.5] * 63]


def test_batch_parsing_builds_hands_and_sorts_by_time():
    frames = parse_batch([_row(at=20), [10, None, 0, None]])
    assert [frame.at_ms for frame in frames] == [10, 20]
    assert frames[0].hand is None
    assert len(frames[1].hand.landmarks) == 21 and frames[1].hand.label == "Right"


@pytest.mark.parametrize(
    "batch",
    [
        "not a list",
        [_row()] * (MAX_FRAMES_PER_BATCH + 1),
        [_row(coordinates=[0.5] * 62)],
        [_row(coordinates=[float("nan")] + [0.5] * 62)],
        [_row(coordinates=[5.0] * 63)],
        [_row(label="Middle")],
        [_row(score=1.5)],
        [_row(at=-1)],
        [_row(at=True)],
        [[1, "Right", 0.9]],
    ],
)
def test_untrusted_batches_are_rejected_whole(batch):
    with pytest.raises(BatchError):
        parse_batch(batch)


# ----------------------------------------------------------- remote access
@pytest.mark.parametrize(
    ("headers", "local"),
    [
        ({"Host": "localhost:8501"}, True),
        ({"host": "127.0.0.1:8501"}, True),
        ({"host": "[::1]:8501"}, True),
        (None, True),
        ({"host": "calm-words-here.trycloudflare.com"}, False),
        ({"host": "localhost:8501", "Cf-Connecting-Ip": "203.0.113.9"}, False),
        ({"host": "localhost:8501", "X-Forwarded-For": "203.0.113.9"}, False),
        ({"host": "192.168.0.10:8501"}, False),
        ({"host": "localhost:8501", "origin": "http://localhost:8501"}, True),
        ({"host": "localhost:8501", "origin": "https://x.trycloudflare.com"}, False),
        ({"host": "127.0.0.1:8501", "Origin": "http://[::1]:8501"}, True),
    ],
)
def test_only_direct_local_requests_reach_the_therapist_panel(headers, local):
    assert is_local_request(headers) is local


def test_tunnel_url_is_parsed_from_cloudflared_output_and_link_built():
    line = "INF |  https://brave-sunny-otter-lake.trycloudflare.com                  |"
    assert TUNNEL_URL.search(line).group(0) == "https://brave-sunny-otter-lake.trycloudflare.com"
    assert patient_link("https://a.trycloudflare.com/", "ABCD-EFGH-JKLM") == (
        "https://a.trycloudflare.com/?sala=ABCD-EFGH-JKLM"
    )


def test_tunnel_points_only_at_the_local_panel(tmp_path):
    command = Tunnel(tmp_path / "cloudflared", 8501).command()
    assert command[-2:] == ["--url", "http://localhost:8501"]
    assert "--no-autoupdate" in command


class FakeTunnel(Tunnel):
    """Runs a Python stand-in that prints what cloudflared prints, to exercise the lifecycle."""

    def __init__(self, script: str) -> None:
        super().__init__(Path(sys.executable), 8501)
        self.script = script

    def command(self) -> list[str]:
        return [sys.executable, "-c", self.script]


def test_tunnel_reports_its_public_url_and_stops():
    tunnel = FakeTunnel(
        "import sys, time\n"
        "print('INF Requesting new quick Tunnel on trycloudflare.com...', file=sys.stderr)\n"
        "print('INF |  https://calm-river-test.trycloudflare.com  |', file=sys.stderr, flush=True)\n"
        "time.sleep(60)\n"
    )
    assert tunnel.start(timeout=20) == "https://calm-river-test.trycloudflare.com"
    assert tunnel.running
    assert tunnel.start() == tunnel.url  # Idempotent while running.
    tunnel.stop()
    assert not tunnel.running and tunnel.url is None


def test_tunnel_that_exits_without_an_address_reports_an_error():
    tunnel = FakeTunnel("import sys; print('ERR failed to connect', file=sys.stderr)")
    with pytest.raises(TunnelError, match=r"remote\.error\.start"):
        tunnel.start(timeout=20)
    assert not tunnel.running


def test_cloudflared_is_found_in_the_tools_directory(tmp_path, monkeypatch):
    import spectra.remote as remote

    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    assert find_cloudflared(tmp_path) is None
    (tmp_path / binary_name()).write_bytes(b"binary")
    assert find_cloudflared(tmp_path) == tmp_path / binary_name()


def test_failed_download_leaves_no_partial_file(tmp_path, monkeypatch):
    import spectra.remote as remote

    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(remote.urllib.request, "urlopen", offline)
    with pytest.raises(TunnelError, match=r"remote\.error\.download"):
        download_cloudflared(tmp_path)
    assert list(tmp_path.iterdir()) == []
