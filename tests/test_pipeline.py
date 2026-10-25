import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

import scripts.generate as generate
import scripts.ingest as ingest
from scripts.document_processor import CHUNK_SIZE, chunk_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FakeVectorstore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def similarity_search(self, query, k=5):
        self.queries.append((query, k))
        return self.chunks[:k]


CHUNKS = [
    Document(page_content="Peak margin applies from Dec 2020.", metadata={"source": "circular.txt"}),
    Document(page_content="Second chunk of the circular.", metadata={"source": "circular.txt"}),
    Document(page_content="STT is 0.1% on delivery.", metadata={"source": "stt.txt"}),
]


@pytest.fixture
def fake_llm(mocker):
    llm = FakeListChatModel(responses=["Dec 2020."])
    mocker.patch.object(generate, "get_llm", return_value=llm)
    return llm


@pytest.fixture
def fake_corpus(mocker):
    store = FakeVectorstore(CHUNKS)
    mocker.patch.object(generate, "get_vectorstore", return_value=store)
    return store


def test_importing_generate_needs_no_api_key_or_index(tmp_path):
    # Run from an empty dir with no GROQ key: import must not build the LLM or load the index
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "GROQ_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.generate"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_generate_answer_uses_default_corpus(fake_llm, fake_corpus):
    answer, chunks = generate.generate_answer("When did peak margin start?")
    assert answer == "Dec 2020."
    assert chunks == CHUNKS
    assert fake_corpus.queries == [("When did peak margin start?", 5)]


def test_generate_answer_prefers_uploaded_vectorstore(fake_llm, fake_corpus):
    uploaded = FakeVectorstore([Document(page_content="My doc.", metadata={"source": "mine.pdf"})])
    _, chunks = generate.generate_answer("question", custom_vectorstore=uploaded)
    assert [c.metadata["source"] for c in chunks] == ["mine.pdf"]
    assert fake_corpus.queries == []


def test_stream_sends_sources_block_then_answer(fake_llm, fake_corpus):
    pieces = list(generate.generate_answer_stream("question"))
    header = pieces[0]
    assert header.startswith("::SOURCES::") and header.endswith("::END_SOURCES::")
    sources = json.loads(header[len("::SOURCES::"):-len("::END_SOURCES::")])
    assert [s["source"] for s in sources] == ["circular.txt", "stt.txt"]
    assert "".join(pieces[1:]) == "Dec 2020."


def test_format_sources_dedupes_by_document_keeping_first_chunk():
    sources = generate.format_sources(CHUNKS)
    assert sources == [
        {"source": "circular.txt", "excerpt": "Peak margin applies from Dec 2020."},
        {"source": "stt.txt", "excerpt": "STT is 0.1% on delivery."},
    ]


def test_format_sources_truncates_excerpt():
    long_chunk = [Document(page_content="x" * 1000, metadata={"source": "a.pdf"})]
    assert len(generate.format_sources(long_chunk)[0]["excerpt"]) == 400


def test_format_sources_handles_missing_source():
    assert generate.format_sources([Document(page_content="t", metadata={})])[0]["source"] == "unknown"


def test_ingest_and_uploads_share_the_same_chunker():
    assert ingest.chunk_documents is chunk_documents


def test_chunk_documents_respects_size_and_keeps_metadata():
    doc = Document(page_content="word " * 1000, metadata={"source": "big.txt"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
    assert all(c.metadata["source"] == "big.txt" for c in chunks)


# --- Conversation memory ---

HISTORY = [
    {"question": "How is STT charged on futures?", "answer": "On the sell side."},
]


@pytest.fixture
def fake_condense(mocker):
    llm = FakeListChatModel(responses=["How is STT charged on options?"])
    mocker.patch.object(generate, "get_condense_llm", return_value=llm)
    return llm


def test_condense_without_history_returns_question_unchanged(mocker):
    condense_llm = mocker.patch.object(generate, "get_condense_llm")
    assert generate.condense_question("What is STT?", []) == "What is STT?"
    condense_llm.assert_not_called()


def test_condense_rewrites_follow_up(fake_condense):
    assert generate.condense_question("what about options?", HISTORY) == "How is STT charged on options?"


def test_condense_falls_back_to_previous_question_on_error(mocker):
    broken = mocker.MagicMock()
    broken.invoke.side_effect = RuntimeError("groq down")
    mocker.patch.object(generate, "get_condense_llm", return_value=broken)
    assert generate.condense_question("what about options?", HISTORY) == (
        "How is STT charged on futures? what about options?"
    )


def test_messages_include_recent_history_in_order():
    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    messages = generate._build_messages("now", CHUNKS, history)
    roles_and_text = [(role, text) for role, text in messages[1:]]
    # Only the last MAX_HISTORY_TURNS turns, oldest first, then the new question
    assert roles_and_text == [
        ("human", "q2"), ("ai", "a2"),
        ("human", "q3"), ("ai", "a3"),
        ("human", "q4"), ("ai", "a4"),
        ("human", "now"),
    ]


def test_follow_up_retrieves_with_standalone_question(fake_llm, fake_corpus, fake_condense):
    generate.generate_answer("what about options?", history=HISTORY)
    assert fake_corpus.queries == [("How is STT charged on options?", 5)]


def test_stream_follow_up_retrieves_with_standalone_question(fake_llm, fake_corpus, fake_condense):
    list(generate.generate_answer_stream("what about options?", history=HISTORY))
    assert fake_corpus.queries == [("How is STT charged on options?", 5)]
