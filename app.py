from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import logging
import os
import tempfile
import threading
import time
import uuid
from langchain_community.vectorstores import FAISS
from pypdf.errors import PyPdfError

from scripts.document_processor import process_uploaded_pdf
from fastapi.responses import StreamingResponse

from scripts.generate import (
    format_sources,
    generate_answer,
    generate_answer_stream,
    get_embeddings,
    get_llm,
    get_vectorstore,
)


@asynccontextmanager
async def lifespan(app):
    # Load the model and index at startup so the first user request isn't slowed down
    get_llm()
    get_vectorstore()
    yield


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
SESSION_TTL_SECONDS = 60 * 60        # drop uploaded docs unused for 1 hour
MAX_SESSIONS = 50                    # hard cap so memory can't grow unbounded

session_vectorstores = {}  # session_id -> {"vectorstore": FAISS, "last_used": float}
sessions_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://frontend-omega-six-90.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


def _evict_expired_sessions():
    now = time.time()
    expired = [
        sid for sid, s in session_vectorstores.items()
        if now - s["last_used"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del session_vectorstores[sid]


def get_session_vectorstore(session_id):
    """Return the uploaded doc's vectorstore, or None when no session is requested.

    Raises 410 if a session was requested but has expired or been lost (e.g. after
    a backend restart), so we never silently answer from the default corpus instead.
    """
    if not session_id:
        return None
    with sessions_lock:
        _evict_expired_sessions()
        session = session_vectorstores.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=410,
                detail="Your uploaded document has expired. Please upload it again.",
            )
        session["last_used"] = time.time()
        return session["vectorstore"]


def store_session_vectorstore(session_id, vectorstore):
    with sessions_lock:
        _evict_expired_sessions()
        if len(session_vectorstores) >= MAX_SESSIONS:
            oldest = min(session_vectorstores, key=lambda sid: session_vectorstores[sid]["last_used"])
            del session_vectorstores[oldest]
        session_vectorstores[session_id] = {"vectorstore": vectorstore, "last_used": time.time()}


@app.post("/query")
def query_endpoint(request: QueryRequest):
    custom_vs = get_session_vectorstore(request.session_id)
    answer, sources = generate_answer(request.question, custom_vectorstore=custom_vs)

    return {
        "answer": answer,
        "sources": format_sources(sources),
    }

@app.post("/query-stream")
def query_stream_endpoint(request: QueryRequest):
    custom_vs = get_session_vectorstore(request.session_id)

    def event_generator():
        for piece in generate_answer_stream(request.question, custom_vectorstore=custom_vs):
            yield piece

    return StreamingResponse(event_generator(), media_type="text/plain")

# Plain `def` (not async) so the blocking PDF parsing and embedding calls run in
# FastAPI's threadpool instead of stalling the event loop for every other request.
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large. The limit is 10 MB.")
    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="This file doesn't look like a valid PDF.")

    # Let tempfile pick the path; the user-supplied filename is only kept as metadata.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    try:
        try:
            chunks = process_uploaded_pdf(temp_path, filename)
        except (PyPdfError, ValueError):
            raise HTTPException(status_code=400, detail="This PDF could not be read. It may be corrupted or encrypted.")

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from this PDF. Scanned/image-only PDFs aren't supported yet.",
            )

        try:
            session_vectorstore = FAISS.from_documents(chunks, get_embeddings())
        except Exception:
            logger.exception("Embedding failed for uploaded file %s", filename)
            raise HTTPException(status_code=502, detail="The embedding service is unavailable. Please try again shortly.")
    finally:
        os.unlink(temp_path)

    session_id = str(uuid.uuid4())
    store_session_vectorstore(session_id, session_vectorstore)

    return {"session_id": session_id, "filename": filename, "chunks": len(chunks)}
