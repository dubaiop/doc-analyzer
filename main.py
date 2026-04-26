import os
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from parsers import parse_document
from llm import ask_llm, get_available_providers

app = FastAPI(title="Doc Analyzer")

# Persist documents to disk so they survive server restarts
STORAGE = Path(os.environ.get("STORAGE_DIR", "/tmp/doc-analyzer"))
STORAGE.mkdir(parents=True, exist_ok=True)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    doc_id: str
    provider: str
    question: str

class ActionRequest(BaseModel):
    doc_id: str
    provider: str
    action: str  # summarize | keypoints | translate


# ── Storage helpers ───────────────────────────────────────────────────────────

def _doc_path(doc_id: str) -> Path:
    return STORAGE / f"{doc_id}.txt"

def _meta_path(doc_id: str) -> Path:
    return STORAGE / f"{doc_id}.meta.json"

def _history_path(doc_id: str) -> Path:
    return STORAGE / f"{doc_id}.history.json"

def _save_text(doc_id: str, text: str):
    _doc_path(doc_id).write_text(text, encoding="utf-8")

def _load_text(doc_id: str) -> str | None:
    p = _doc_path(doc_id)
    return p.read_text(encoding="utf-8") if p.exists() else None

def _save_meta(doc_id: str, meta: dict):
    _meta_path(doc_id).write_text(json.dumps(meta), encoding="utf-8")

def _load_meta(doc_id: str) -> dict | None:
    p = _meta_path(doc_id)
    return json.loads(p.read_text()) if p.exists() else None

def _save_history(doc_id: str, history: list):
    _history_path(doc_id).write_text(json.dumps(history), encoding="utf-8")

def _load_history(doc_id: str) -> list:
    p = _history_path(doc_id)
    return json.loads(p.read_text()) if p.exists() else []

def _delete_doc_files(doc_id: str):
    for p in [_doc_path(doc_id), _meta_path(doc_id), _history_path(doc_id)]:
        p.unlink(missing_ok=True)

def _list_all_meta() -> list[dict]:
    return [
        json.loads(p.read_text())
        for p in STORAGE.glob("*.meta.json")
    ]


# ── Provider ──────────────────────────────────────────────────────────────────

@app.get("/api/providers")
def providers():
    return get_available_providers()


# ── Documents ─────────────────────────────────────────────────────────────────

@app.get("/api/documents")
def list_documents():
    return _list_all_meta()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")
    try:
        text = parse_document(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    doc_id = file.filename.replace(" ", "_")
    meta = {
        "doc_id": doc_id,
        "name": file.filename,
        "chars": len(text),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    _save_text(doc_id, text)
    _save_meta(doc_id, meta)
    _save_history(doc_id, [])

    preview = text[:300] + ("..." if len(text) > 300 else "")
    return {"doc_id": doc_id, "chars": len(text), "preview": preview}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    _delete_doc_files(doc_id)
    return {"ok": True}


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history/{doc_id}")
def get_history(doc_id: str):
    return _load_history(doc_id)


@app.post("/api/clear-history/{doc_id}")
def clear_history(doc_id: str):
    _save_history(doc_id, [])
    return {"ok": True}


# ── Analysis ──────────────────────────────────────────────────────────────────

@app.post("/api/action")
def run_action(req: ActionRequest):
    text = _get_doc(req.doc_id)
    prompts = {
        "summarize": (
            "You are a document analyst. Summarize the document clearly and concisely.",
            f"Please summarize this document:\n\n{text[:8000]}",
        ),
        "keypoints": (
            "You are a document analyst. Extract the most important key points.",
            f"Extract the key points from this document as a numbered list:\n\n{text[:8000]}",
        ),
        "translate": (
            "You are a translator. Translate the document to English if it is not already in English. "
            "If it is already in English, provide a French translation.",
            f"Translate this document:\n\n{text[:8000]}",
        ),
    }
    if req.action not in prompts:
        raise HTTPException(status_code=400, detail="Unknown action")

    system, user_msg = prompts[req.action]
    try:
        result = ask_llm(req.provider, system, [{"role": "user", "content": user_msg}])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    history = _load_history(req.doc_id)
    history.append({"role": "user", "content": f"[{req.action.capitalize()}]"})
    history.append({"role": "assistant", "content": result})
    _save_history(req.doc_id, history)
    return {"result": result}


@app.post("/api/chat")
def chat(req: ChatRequest):
    text = _get_doc(req.doc_id)
    system = (
        "You are a helpful document assistant. Answer questions strictly based on the document below. "
        "If the answer is not in the document, say so clearly.\n\n"
        f"Document:\n{text[:8000]}"
    )

    history = _load_history(req.doc_id)
    messages = history + [{"role": "user", "content": req.question}]

    try:
        answer = ask_llm(req.provider, system, messages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})
    _save_history(req.doc_id, history)
    return {"answer": answer}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_doc(doc_id: str) -> str:
    text = _load_text(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="Document not found. Please upload it again.")
    return text


app.mount("/", StaticFiles(directory="static", html=True), name="static")
