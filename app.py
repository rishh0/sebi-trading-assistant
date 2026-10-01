from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scripts.generate import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_endpoint(request: QueryRequest):
    answer, sources = generate_answer(request.question)

    unique_sources = sorted(set(s.metadata.get("source", "unknown") for s in sources))

    return {
        "answer": answer,
        "sources": unique_sources,
    }