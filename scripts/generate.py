import json
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

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index"

SYSTEM_PROMPT_TEMPLATE = """You are a compliance assistant that answers questions about Indian securities trading regulations and taxation, using ONLY the context provided below.

Rules:
- Answer only using the information in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say clearly: "I don't have enough information in my documents to answer this confidently — please verify with a CA or your broker."
- Do not guess or make up numbers, rates, or rules that are not explicitly stated in the context.
- Keep your answer concise and direct.

Context:
{context}"""


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
def get_embeddings():
    return LocalEmbeddings()


@lru_cache(maxsize=1)
def get_vectorstore():
    return FAISS.load_local(
        str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


def retrieve_chunks(query, k=5, custom_vectorstore=None):
    vectorstore = custom_vectorstore or get_vectorstore()
    return vectorstore.similarity_search(query, k=k)


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


def _build_messages(question, chunks):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=_build_context(chunks))
    return [
        ("system", system_prompt),
        ("human", question),
    ]


def generate_answer(question, custom_vectorstore=None):
    chunks = retrieve_chunks(question, custom_vectorstore=custom_vectorstore)
    response = get_llm().invoke(_build_messages(question, chunks))
    return response.content, chunks


def generate_answer_stream(question, custom_vectorstore=None):
    chunks = retrieve_chunks(question, custom_vectorstore=custom_vectorstore)

    yield "::SOURCES::" + json.dumps(format_sources(chunks)) + "::END_SOURCES::"

    for token in get_llm().stream(_build_messages(question, chunks)):
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
