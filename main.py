import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from pdf_ingest import get_store
from graph import graph

app = FastAPI(title="AI Research Assistant API")

store = get_store()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── request/response models ────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    user_id: str
    session_id: str
    doc_id: Optional[str] = None


class QueryResponse(BaseModel):
    final_report: str


#upload endpoint
@app.post("/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    """Saves the uploaded PDF, ingests it into the vectorstore, returns doc_id."""
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id = store.ingest(str(file_path), user_id=user_id, filename=file.filename)

    return {"doc_id": doc_id, "filename": file.filename}


# ── list user's documents ───────────────────────────────────────────
@app.get("/documents/{user_id}")
async def get_user_documents(user_id: str):
    docs = store.list_user_docs(user_id)
    return {"documents": docs}


# ── delete a document ────────────────────────────────────────────────
@app.delete("/documents/{user_id}/{doc_id}")
async def delete_document(user_id: str, doc_id: str):
    store.delete_doc(user_id, doc_id)
    return {"status": "deleted", "doc_id": doc_id}


# ── query endpoint — runs the full graph ─────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = graph.invoke(
        {
            "query": request.query,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "doc_id": request.doc_id,
        },
        config={"configurable": {"thread_id": request.session_id}},
    )
    return {"final_report": result["final_report"]}


# ── health check ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}