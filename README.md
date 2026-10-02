# SEBI Trading Compliance Assistant

A RAG-based assistant that answers questions about Indian securities trading regulations (SEBI margin rules) and F&O taxation, grounded in real regulatory circulars and tax reference material — not the model's general knowledge.

**Live demo:** https://frontend-omega-six-90.vercel.app
**Backend API:** https://sebi-trading-assistant.onrender.com

> Note: the backend runs on Render's free tier, which spins down after inactivity. The first request after a period of no traffic may take 30-60 seconds to respond while it wakes up.

## The problem

Indian retail traders need to navigate margin rules, F&O taxation treatment, STT rates, and tax audit thresholds — information that exists in official SEBI circulars and tax guidance, but is scattered across dense PDFs and rarely explained together. This assistant answers questions directly from a curated set of these documents, citing its sources, and explicitly declines to answer when a question falls outside what its documents cover — rather than guessing.

## Architecture

- **Frontend:** Next.js, deployed on Vercel
- **Backend:** FastAPI, deployed on Render
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, called via Hugging Face's Inference API in production (kept local-only for the one-time document ingestion step)
- **Vector store:** FAISS (local index, committed to the repo)
- **LLM:** Llama 3.3 70B via Groq, through LangChain's chat-model interface
- **Orchestration:** LangChain (document loading, chunking, retrieval, prompt construction)

## Key design decisions

- **Guardrail-first prompting:** the system prompt explicitly instructs the model to answer only from retrieved context and to decline when the context is insufficient, rather than guessing. This was deliberately tested (see Evaluation below) against both thin-context and fully out-of-scope questions.
- **Source attribution:** every answer is returned alongside the specific document(s) it was grounded in, so a user can verify the answer rather than trust it blindly.
- **Scoped corpus as a deliberate MVP decision:** the knowledge base currently covers 7 documents (SEBI peak margin circulars, F&O taxation rules, tax audit thresholds, STT rates) rather than the full regulatory corpus. The retrieval/generation pipeline is corpus-agnostic and would scale to a larger document set without architectural changes — this scope was chosen to validate the approach end-to-end rather than to be comprehensive.

## Evaluation

Measured on a hand-labeled, 20-question test set spanning all 7 source documents plus deliberately out-of-scope questions:

| Metric | Result |
|---|---|
| Fully correct | 18/20 (90%) |
| Partially correct (right facts, minor omission) | 2/20 (10%) |
| Incorrect / hallucinated | 0/20 (0%) |
| Out-of-scope questions correctly declined | 4/4 (100%) |

Full question set and results: [`eval/eval_questions.json`](./eval/eval_questions.json), [`eval/scorecard.md`](./eval/scorecard.md)

## Known limitations

- Free-tier backend hosting means a cold-start delay after inactivity (see note above)
- Document corpus is intentionally scoped to 7 documents (see Key design decisions)
- Out-of-scope questions all currently return the same decline message rather than a reason-specific one (e.g. "outside domain" vs. "retrieved but insufficient detail") — a natural next improvement
- No conversation memory — each question is answered independently

## Running locally

**Backend:**
```bash
pip install -r requirements.txt
# add GROQ_API_KEY and HF_TOKEN to a .env file
python scripts/ingest.py   # one-time: build the vector index from docs/
uvicorn app:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
# add NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 to .env.local
npm run dev
```

## Tech stack

Python, FastAPI, LangChain, FAISS, Hugging Face Inference API, Groq, Next.js, React