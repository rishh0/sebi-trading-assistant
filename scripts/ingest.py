"""
scripts/ingest.py

Reads every PDF/TXT file in docs/, splits it into chunks, generates
embeddings locally (free, no API key) using fastembed,
and saves a FAISS vector store to data/faiss_index/ for later retrieval.

Run with: python scripts/ingest.py
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

sys.path.append(str(Path(__file__).parent.parent))  # allow `python scripts/<name>.py`
from scripts.document_processor import chunk_documents
from scripts.embeddings import LocalEmbeddings

DOCS_DIR = Path("docs")
INDEX_DIR = Path("data/faiss_index")


def load_documents():
    """Load every .pdf and .txt file in docs/ as LangChain Document objects."""
    documents = []

    for file_path in DOCS_DIR.iterdir():
        if file_path.suffix.lower() == ".pdf":
            print(f"Loading PDF: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".txt":
            print(f"Loading TXT: {file_path.name}")
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            continue  # skip anything that isn't a pdf/txt

        docs = loader.load()
        # tag every chunk from this file with a clean source name,
        # so we can show "which document" an answer came from later
        for doc in docs:
            doc.metadata["source"] = file_path.name
        documents.extend(docs)

    return documents


def main():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        print(f"No files found in {DOCS_DIR}/. Add your PDFs/TXTs first.")
        return

    print("Step 1/4: Loading documents...")
    documents = load_documents()
    print(f"  -> loaded {len(documents)} document section(s)\n")

    print("Step 2/4: Chunking...")
    chunks = chunk_documents(documents)
    print(f"  -> created {len(chunks)} chunks\n")

    print("Step 3/4: Loading embedding model (first run downloads it, then cached)...")
    embeddings = LocalEmbeddings()
    print("  -> model ready\n")

    print("Step 4/4: Generating embeddings + building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print(f"\nDone. Saved FAISS index with {len(chunks)} chunks to {INDEX_DIR}/")


if __name__ == "__main__":
    main()