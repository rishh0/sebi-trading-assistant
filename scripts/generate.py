import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

INDEX_DIR = Path("data/faiss_index")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=512,
    temperature=0.1,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
)

def retrieve_chunks(query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    return results

def generate_answer(question):
    chunks = retrieve_chunks(question)

    context_text = ""
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        context_text += f"[Source: {source}]\n{chunk.page_content}\n\n"

    system_prompt = """You are a compliance assistant that answers questions about Indian securities trading regulations and taxation, using ONLY the context provided below.

Rules:
- Answer only using the information in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say clearly: "I don't have enough information in my documents to answer this confidently — please verify with a CA or your broker."
- Do not guess or make up numbers, rates, or rules that are not explicitly stated in the context.
- Keep your answer concise and direct.

Context:
""" + context_text
    
    response = llm.invoke([
        ("system", system_prompt),
        ("human", question),
    ])

    return response.content, chunks


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