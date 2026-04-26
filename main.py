import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from parsers import parse_document
from llm import ask_llm, get_available_providers

app = FastAPI(title="Doc Analyzer")

# doc_id -> raw text
_documents: dict[str, str] = {}
# doc_id -> metadata
_doc_meta: dict[str, dict] = {}
# doc_id -> list of {"role": "user"/"assistant", "content": "..."}
_histories: dict[str, list[dict]] = {}


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    doc_id: str
    provider: str
    question: str

class ActionRequest(BaseModel):
    doc_id: str
    provider: str
    action: str  # summarize | keypoints | translate


# ── Provider ──────────────────────────────────────────────────────────────────

@app.get("/api/providers")
def providers():
    return get_available_providers()


# ── Documents ─────────────────────────────────────────────────────────────────

@app.get("/api/documents")
def list_documents():
    return list(_doc_meta.values())


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
    _documents[doc_id] = text
    _histories[doc_id] = []
    _doc_meta[doc_id] = {
        "doc_id": doc_id,
        "name": file.filename,
        "chars": len(text),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    preview = text[:300] + ("..." if len(text) > 300 else "")
    return {"doc_id": doc_id, "chars": len(text), "preview": preview}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    _documents.pop(doc_id, None)
    _histories.pop(doc_id, None)
    _doc_meta.pop(doc_id, None)
    return {"ok": True}


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history/{doc_id}")
def get_history(doc_id: str):
    return _histories.get(doc_id, [])


@app.post("/api/clear-history/{doc_id}")
def clear_history(doc_id: str):
    _histories[doc_id] = []
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save to history
    _histories[req.doc_id].append({"role": "user", "content": f"[{req.action.capitalize()}]"})
    _histories[req.doc_id].append({"role": "assistant", "content": result})
    return {"result": result}


@app.post("/api/chat")
def chat(req: ChatRequest):
    text = _get_doc(req.doc_id)
    system = (
        "You are a helpful document assistant. Answer questions strictly based on the document below. "
        "If the answer is not in the document, say so clearly.\n\n"
        f"Document:\n{text[:8000]}"
    )

    history = _histories.get(req.doc_id, [])
    # Build messages: history + new question
    messages = history + [{"role": "user", "content": req.question}]

    try:
        answer = ask_llm(req.provider, system, messages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Persist to history
    _histories[req.doc_id].append({"role": "user", "content": req.question})
    _histories[req.doc_id].append({"role": "assistant", "content": answer})
    return {"answer": answer}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_doc(doc_id: str) -> str:
    text = _documents.get(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="Document not found. Please upload it again.")
    return text


app.mount("/", StaticFiles(directory="static", html=True), name="static")
