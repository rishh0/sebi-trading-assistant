from langchain_core.documents import Document
from scripts.generate import _build_context, SYSTEM_PROMPT_TEMPLATE

def test_build_context_includes_source_and_text():
    chunks = [Document(page_content="Rule A applies.", metadata={"source": "doc1.pdf"})]
    context = _build_context(chunks)
    assert "[Source: doc1.pdf]" in context
    assert "Rule A applies." in context


def test_build_context_keeps_chunks_in_order():
    chunks = [
        Document(page_content="First chunk", metadata={"source": "a.pdf"}),
        Document(page_content="Second chunk", metadata={"source": "b.pdf"}),
    ]
    context = _build_context(chunks)
    assert context.index("First chunk") < context.index("Second chunk")


def test_build_context_uses_unknown_when_source_missing():
    chunks = [Document(page_content="No source here", metadata={})]
    assert "[Source: unknown]" in _build_context(chunks)


def test_build_context_empty_list_returns_empty_string():
    assert _build_context([]) == ""


def test_system_prompt_contains_guardrail_instructions():
    assert "ONLY the context" in SYSTEM_PROMPT_TEMPLATE
    assert "I don't have enough information in my documents" in SYSTEM_PROMPT_TEMPLATE


def test_system_prompt_template_inserts_context():
    prompt = SYSTEM_PROMPT_TEMPLATE.format(context="some retrieved text")
    assert "some retrieved text" in prompt