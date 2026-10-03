from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_uploaded_pdf(file_path, original_filename):
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = original_filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks