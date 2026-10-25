from pathlib import Path

import pytest

from scripts.document_processor import process_uploaded_pdf

TEST_PDF = Path("docs/02_BOD_Margin_Ammendment.pdf")
ORIGINAL_NAME = "my_uploaded_file.pdf"


@pytest.fixture(scope="module")
def chunks():
    return process_uploaded_pdf(TEST_PDF, ORIGINAL_NAME)


def test_returns_at_least_one_chunk(chunks):
    assert len(chunks) > 0


def test_every_chunk_is_tagged_with_original_filename(chunks):
    for chunk in chunks:
        assert chunk.metadata["source"] == ORIGINAL_NAME


def test_no_chunk_exceeds_size_limit(chunks):
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000


def test_no_chunk_is_empty(chunks):
    for chunk in chunks:
        assert chunk.page_content.strip() != ""