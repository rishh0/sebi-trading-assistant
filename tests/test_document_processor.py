from pathlib import Path

from scripts.document_processor import process_uploaded_pdf

TEST_PDF = Path("docs/02_BOD_Margin_Ammendment.pdf")


def test_process_uploaded_pdf_returns_chunks():
    chunks = process_uploaded_pdf(TEST_PDF, "my_uploaded_file.pdf")

    assert len(chunks) > 0