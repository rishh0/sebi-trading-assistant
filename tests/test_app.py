from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)

REAL_PDF = Path("docs/02_BOD_Margin_Ammendment.pdf")


@pytest.fixture(autouse=True)
def clear_sessions():
    app_module.session_vectorstores.clear()
    yield
    app_module.session_vectorstores.clear()


@pytest.fixture
def fake_faiss(mocker):
    # Avoid calling the HF embedding API; we only care about upload handling here
    return mocker.patch.object(app_module.FAISS, "from_documents", return_value="fake-vectorstore")


def upload(name, content):
    return client.post("/upload", files={"file": (name, content, "application/pdf")})


def test_upload_valid_pdf_creates_session(fake_faiss):
    res = upload("my_doc.pdf", REAL_PDF.read_bytes())
    assert res.status_code == 200
    body = res.json()
    assert body["filename"] == "my_doc.pdf"
    assert body["chunks"] > 0
    assert body["session_id"] in app_module.session_vectorstores


def test_upload_rejects_non_pdf_extension(fake_faiss):
    res = upload("notes.txt", b"%PDF-1.4 hello")
    assert res.status_code == 400
    fake_faiss.assert_not_called()


def test_upload_rejects_file_without_pdf_header(fake_faiss):
    res = upload("fake.pdf", b"not really a pdf")
    assert res.status_code == 400
    fake_faiss.assert_not_called()


def test_upload_rejects_oversized_file(fake_faiss, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 100)
    res = upload("big.pdf", b"%PDF" + b"x" * 200)
    assert res.status_code == 413


def test_upload_corrupt_pdf_returns_400(fake_faiss):
    res = upload("broken.pdf", b"%PDF-1.4 garbage with no structure")
    assert res.status_code == 400


def test_upload_pdf_without_text_returns_400(fake_faiss, tmp_path):
    from pypdf import PdfWriter

    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    writer.write(blank)

    res = upload("blank.pdf", blank.read_bytes())
    assert res.status_code == 400
    assert "No text" in res.json()["detail"]


def test_upload_strips_directory_from_filename(fake_faiss):
    res = upload("../../etc/evil.pdf", REAL_PDF.read_bytes())
    assert res.status_code == 200
    assert res.json()["filename"] == "evil.pdf"


def test_unknown_session_returns_410():
    res = client.post("/query-stream", json={"question": "hi", "session_id": "does-not-exist"})
    assert res.status_code == 410


def test_expired_sessions_are_evicted(monkeypatch):
    app_module.store_session_vectorstore("old", "vs")
    app_module.session_vectorstores["old"]["last_used"] -= app_module.SESSION_TTL_SECONDS + 1

    with pytest.raises(app_module.HTTPException) as exc:
        app_module.get_session_vectorstore("old")
    assert exc.value.status_code == 410
    assert "old" not in app_module.session_vectorstores


def test_session_cap_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_SESSIONS", 2)
    app_module.store_session_vectorstore("a", "vs-a")
    app_module.store_session_vectorstore("b", "vs-b")
    app_module.get_session_vectorstore("a")  # touch "a" so "b" is the oldest
    app_module.session_vectorstores["b"]["last_used"] -= 10

    app_module.store_session_vectorstore("c", "vs-c")

    assert set(app_module.session_vectorstores) == {"a", "c"}
