import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

sys.path.append(str(Path(__file__).parent.parent))  # allow `python scripts/<name>.py`
from scripts.embeddings import LocalEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index"

SYSTEM_PROMPT_TEMPLATE = """You are a compliance assistant that answers questions about Indian securities trading regulations and taxation, using ONLY the context provided below.

Rules:
- Answer only using the information in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say clearly: "I don't have enough information in my documents to answer this confidently — please verify with a CA or your broker."
- Do not guess or make up numbers, rates, or rules that are not explicitly stated in the context.
- Keep your answer concise and direct.

Context:
{context}"""

# Retrieval distance (squared L2 on normalized embeddings; lower = closer) above
# which a question is treated as outside the default corpus. Calibrated on the eval
# set plus extra probes: answerable questions scored <= 1.14, clearly unrelated ones
# (weather, cricket, laptops, RBI policy...) >= 1.26. Borderline questions below the
# cutoff still reach the LLM, whose prompt makes it decline when context is thin.
OUT_OF_SCOPE_DISTANCE = 1.25

OUT_OF_SCOPE_MESSAGE = (
    "That's outside what I can help with. I answer questions about **SEBI margin rules** "
    "(peak margin, upfront margin collection), **F&O taxation**, **tax audits under "
    "Section 44AB**, and **STT**, based on official circulars and tax reference notes. "
    "Try asking about one of those, or upload your own PDF to ask about it."
)

# How many previous question/answer pairs are used for follow-up questions
MAX_HISTORY_TURNS = 3

CONDENSE_PROMPT_TEMPLATE = """Rewrite the user's latest message as a standalone question that can be understood without the conversation. Keep it in the same language and domain; don't answer it. If it is already standalone, return it unchanged. Output only the question.

Conversation:
{conversation}

Latest message: {question}"""


# The LLM client, embedding model and index are created on first use rather than
# at import, so importing this module (e.g. in tests) needs no API key or index.
@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1024,
        temperature=0.1,
    )


@lru_cache(maxsize=1)
def get_condense_llm():
    # Rewriting a follow-up is a small task: low reasoning effort keeps it ~0.5s
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=512,
        temperature=0,
        reasoning_effort="low",
        reasoning_format="hidden",
    )


@lru_cache(maxsize=1)
def get_embeddings():
    return LocalEmbeddings()


@lru_cache(maxsize=1)
def get_vectorstore():
    return FAISS.load_local(
        str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


def _recent(history):
    return list(history or [])[-MAX_HISTORY_TURNS:]


def condense_question(question, history):
    """Turn a follow-up like "what about options?" into a standalone question for retrieval."""
    history = _recent(history)
    if not history:
        return question

    conversation = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in history
    )
    try:
        response = get_condense_llm().invoke(
            CONDENSE_PROMPT_TEMPLATE.format(conversation=conversation, question=question)
        )
        standalone = response.content.strip()
    except Exception:
        logger.exception("Question rewrite failed; falling back to concatenation")
        standalone = ""

    # Fallback still gives retrieval the topic of the previous question
    return standalone or f"{history[-1]['question']} {question}"


def retrieve_chunks(query, k=5, custom_vectorstore=None):
    """Returns (chunks, out_of_scope).

    The out-of-scope check only applies to the default corpus: the cutoff was
    calibrated on it, and an uploaded PDF could be about anything.
    """
    if custom_vectorstore is not None:
        return custom_vectorstore.similarity_search(query, k=k), False

    results = get_vectorstore().similarity_search_with_score(query, k=k)
    if not results or results[0][1] > OUT_OF_SCOPE_DISTANCE:
        return [], True
    return [doc for doc, _ in results], False


def format_sources(chunks, excerpt_chars=400):
    """One entry per source document (first matching chunk wins), in retrieval order."""
    seen = set()
    source_details = []
    for chunk in chunks:
        name = chunk.metadata.get("source", "unknown")
        if name not in seen:
            seen.add(name)
            source_details.append({
                "source": name,
                "excerpt": chunk.page_content[:excerpt_chars],
            })
    return source_details


def _build_context(chunks):
    context_text = ""
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        context_text += f"[Source: {source}]\n{chunk.page_content}\n\n"
    return context_text


def _build_messages(question, chunks, history=None):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=_build_context(chunks))
    messages = [("system", system_prompt)]
    for turn in _recent(history):
        messages.append(("human", turn["question"]))
        messages.append(("ai", turn["answer"]))
    messages.append(("human", question))
    return messages


def generate_answer(question, custom_vectorstore=None, history=None):
    search_query = condense_question(question, history)
    chunks, out_of_scope = retrieve_chunks(search_query, custom_vectorstore=custom_vectorstore)
    if out_of_scope:
        return OUT_OF_SCOPE_MESSAGE, []

    response = get_llm().invoke(_build_messages(question, chunks, history))
    return response.content, chunks


def generate_answer_stream(question, custom_vectorstore=None, history=None):
    search_query = condense_question(question, history)
    chunks, out_of_scope = retrieve_chunks(search_query, custom_vectorstore=custom_vectorstore)

    yield "::SOURCES::" + json.dumps(format_sources(chunks)) + "::END_SOURCES::"

    if out_of_scope:
        yield OUT_OF_SCOPE_MESSAGE
        return

    for token in get_llm().stream(_build_messages(question, chunks, history)):
        yield token.content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    if not question:
        print('Usage: python scripts/generate.py "your question"')
    else:
        answer, sources = generate_answer(question)
        print(f"\nQuestion: {question}\n")
        print(f"Answer:\n{answer}\n")
        print("Sources used:")
        unique_sources = sorted(set(s.metadata.get("source", "unknown") for s in sources))
        for src in unique_sources:
            print(f"  - {src}")
