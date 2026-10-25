"""
scripts/embeddings.py

Single embedding implementation shared by ingestion, querying and the API, so
the vectors in the FAISS index always match the vectors used at query time.

Runs all-MiniLM-L6-v2 locally via fastembed (ONNX Runtime, no PyTorch), so it's
free, needs no API key, and fits in a small (512 MB) server.
"""

from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Keep the model inside the project (gitignored) rather than the system temp dir,
# so a model downloaded during the deploy build is still there at runtime.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fastembed_cache"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name=MODEL_NAME):
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(CACHE_DIR))

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.embed(texts)]


if __name__ == "__main__":
    # Used at deploy build time to download the model ahead of the first request:
    #   python -m scripts.embeddings
    LocalEmbeddings().embed_query("warm up")
    print(f"Embedding model ready in {CACHE_DIR}")
