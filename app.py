from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import shutil
import uuid
from langchain_community.vectorstores import FAISS
from pathlib import Path

from scripts.document_processor import process_uploaded_pdf
from scripts.generate import embeddings  # reuse the same embeddings instance

from fastapi.responses import StreamingResponse

from scripts.generate import generate_answer, generate_answer_stream

app = FastAPI()

session_vectorstores = {}  # session_id -> FAISS vectorstore

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


@app.post("/query")
def query_endpoint(request: QueryRequest):
    custom_vs = session_vectorstores.get(request.session_id) if request.session_id else None
    answer, sources = generate_answer(request.question, custom_vectorstore=custom_vs)

    seen = set()
    source_details = []
    for chunk in sources:
        name = chunk.metadata.get("source", "unknown")
        if name not in seen:
            seen.add(name)
            source_details.append({
                "source": name,
                "excerpt": chunk.page_content[:400],
            })

    return {
        "answer": answer,
        "sources": source_details,
    }

@app.post("/query-stream")
def query_stream_endpoint(request: QueryRequest):
    custom_vs = session_vectorstores.get(request.session_id) if request.session_id else None

    def event_generator():
        for piece in generate_answer_stream(request.question, custom_vectorstore=custom_vs):
            yield piece

    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    temp_path = Path(f"/tmp/{session_id}_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks = process_uploaded_pdf(temp_path, file.filename)
        session_vectorstore = FAISS.from_documents(chunks, embeddings)
        session_vectorstores[session_id] = session_vectorstore
    finally:
        temp_path.unlink(missing_ok=True)

    return {"session_id": session_id, "filename": file.filename, "chunks": len(chunks)}