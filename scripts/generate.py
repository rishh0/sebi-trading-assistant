import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

class LightweightHFEmbeddings(Embeddings):
    def __init__(self, model_name, hf_token):
        self.client = InferenceClient(model=model_name, token=hf_token)

    def embed_query(self, text):
        result = self.client.feature_extraction(text)
        return result.mean(axis=0).tolist() if result.ndim > 1 else result.tolist()

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

load_dotenv()

INDEX_DIR = Path("data/faiss_index")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=1024,
    temperature=0.1,
)

embeddings = LightweightHFEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    hf_token=os.getenv("HF_TOKEN"),
)
vectorstore = FAISS.load_local(
    str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
)

SYSTEM_PROMPT_TEMPLATE = """You are a compliance assistant that answers questions about Indian securities trading regulations and taxation, using ONLY the context provided below.

Rules:
- Answer only using the information in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say clearly: "I don't have enough information in my documents to answer this confidently — please verify with a CA or your broker."
- Do not guess or make up numbers, rates, or rules that are not explicitly stated in the context.
- Keep your answer concise and direct.

Context:
{context}"""


def retrieve_chunks(query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    return results


def _build_context(chunks):
    context_text = ""
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        context_text += f"[Source: {source}]\n{chunk.page_content}\n\n"
    return context_text


def generate_answer(question, custom_vectorstore=None):
    if custom_vectorstore:
        chunks = custom_vectorstore.similarity_search(question, k=5)
    else:
        chunks = retrieve_chunks(question)

    context_text = _build_context(chunks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)

    response = llm.invoke([
        ("system", system_prompt),
        ("human", question),
    ])

    return response.content, chunks


def generate_answer_stream(question, custom_vectorstore=None):
    if custom_vectorstore:
        chunks = custom_vectorstore.similarity_search(question, k=5)
    else:
        chunks = retrieve_chunks(question)

    seen = set()
    source_details = []
    for chunk in chunks:
        name = chunk.metadata.get("source", "unknown")
        if name not in seen:
            seen.add(name)
            source_details.append({
                "source": name,
                "excerpt": chunk.page_content[:400],
            })

    import json
    yield "::SOURCES::" + json.dumps(source_details) + "::END_SOURCES::"

    context_text = _build_context(chunks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)

    for token in llm.stream([
        ("system", system_prompt),
        ("human", question),
    ]):
        yield token.content


if __name__ == "__main__":
    import sys
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