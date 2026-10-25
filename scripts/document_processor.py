from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Shared by ingestion (default corpus) and uploads, so both are chunked identically
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # overlap so context isn't lost at boundaries
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # prefer breaking on paragraphs/sentences


def chunk_documents(documents):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    return splitter.split_documents(documents)


def process_uploaded_pdf(file_path, original_filename):
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = original_filename

    return chunk_documents(docs)
