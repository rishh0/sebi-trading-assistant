"""
scripts/query.py

Quick CLI to sanity-check retrieval before we wire up the LLM/API/UI.

Run with: python scripts/query.py "what is the peak margin penalty"
"""

import sys
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = Path("data/faiss_index")


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/query.py "your question here"')
        return

    query = " ".join(sys.argv[1:])

    if not INDEX_DIR.exists():
        print(f"No index found at {INDEX_DIR}/. Run scripts/ingest.py first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search_with_score(query, k=5)

    print(f'\nTop matches for: "{query}"\n')
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:200]
        print(f"{i}. [distance: {score:.3f}] ({source})")
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()