import os
import json
import re
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from parsers import parse_document
from llm import ask_llm, get_available_providers, analyze_video_with_gemini, analyze_video_with_openrouter

app = FastAPI(title="Doc Analyzer")

STORAGE = Path(os.environ.get("STORAGE_DIR", "/tmp/doc-analyzer"))
STORAGE.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    doc_id: str
    provider: str
    question: str

class ActionRequest(BaseModel):
    doc_id: str
    provider: str
    action: str  # summarize | keypoints | translate

class VideoAnalyzeRequest(BaseModel):
    doc_id: str
    prompt: str = "Analyze this video professionally. Describe what is happening, identify key moments, and provide a detailed summary."
    vision_provider: str = "gemini"  # gemini | openrouter

class VideoGenRequest(BaseModel):
    doc_id: str
    provider: str
    lang: str = "en"

# in-memory generation status (keyed by doc_id)
_gen_status: dict[str, dict] = {}


# ── Storage helpers ───────────────────────────────────────────────────────────

def _doc_path(doc_id: str) -> Path:
    return STORAGE / f"{doc_id}.txt"

def _frames_path(doc_id: str) -> Path:
    return STORAGE / f"{doc_id}.frames.json"

def _meta_path(doc_id: str) -> Path:
    return STORAGE / f"{doc_id}.meta.json"

def _history_path(doc_id: str) -> Path:
    return STORAGE / f"{doc_id}.history.json"

def _save_text(doc_id: str, text: str):
    _doc_path(doc_id).write_text(text, encoding="utf-8")

def _load_text(doc_id: str) -> str | None:
    p = _doc_path(doc_id)
    return p.read_text(encoding="utf-8") if p.exists() else None

def _save_frames(doc_id: str, frames: list):
    _frames_path(doc_id).write_text(json.dumps(frames), encoding="utf-8")

def _load_frames(doc_id: str) -> list:
    p = _frames_path(doc_id)
    return json.loads(p.read_text()) if p.exists() else []

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
    for p in [_doc_path(doc_id), _meta_path(doc_id), _history_path(doc_id), _frames_path(doc_id)]:
        p.unlink(missing_ok=True)

def _list_all_meta() -> list[dict]:
    return [json.loads(p.read_text()) for p in STORAGE.glob("*.meta.json")]


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
    ext = Path(file.filename).suffix.lower()

    # ── Video upload ──────────────────────────────────────────────────────────
    if ext in VIDEO_EXTS:
        if len(content) > 500 * 1024 * 1024:  # 500 MB limit for video
            raise HTTPException(status_code=400, detail="Video too large (max 500 MB)")
        try:
            from video_parser import extract_video_frames
            frames = extract_video_frames(content, num_frames=6)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not extract frames: {e}")

        doc_id = file.filename.replace(" ", "_")
        meta = {
            "doc_id": doc_id,
            "name": file.filename,
            "type": "video",
            "frames": len(frames),
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        _save_frames(doc_id, frames)
        _save_text(doc_id, f"[Video: {file.filename}]")
        _save_meta(doc_id, meta)
        _save_history(doc_id, [])
        return {"doc_id": doc_id, "type": "video", "frames": len(frames)}

    # ── Document upload ───────────────────────────────────────────────────────
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
        "type": "document",
        "chars": len(text),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    _save_text(doc_id, text)
    _save_meta(doc_id, meta)
    _save_history(doc_id, [])
    preview = text[:300] + ("..." if len(text) > 300 else "")
    return {"doc_id": doc_id, "type": "document", "chars": len(text), "preview": preview}


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


# ── Video analysis ────────────────────────────────────────────────────────────

@app.post("/api/video-analyze")
def video_analyze(req: VideoAnalyzeRequest):
    frames = _load_frames(req.doc_id)
    if not frames:
        raise HTTPException(status_code=404, detail="No frames found. Please re-upload the video.")
    try:
        if req.vision_provider == "openrouter":
            result = analyze_video_with_openrouter(frames, req.prompt)
        else:
            result = analyze_video_with_gemini(frames, req.prompt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    history = _load_history(req.doc_id)
    history.append({"role": "user", "content": req.prompt})
    history.append({"role": "assistant", "content": result})
    _save_history(req.doc_id, history)
    return {"result": result}


# ── Document analysis ─────────────────────────────────────────────────────────

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


# ── Video generation ──────────────────────────────────────────────────────────

def _do_generate_video(doc_id: str, provider: str, text: str, lang: str):
    from video_generator import build_video

    system = (
        "You are a presentation scriptwriter. "
        "Return ONLY a valid JSON object — no markdown, no code fences, no extra text."
    )
    user_msg = (
        "Create exactly 5 slides for a professional video presentation from the document below. "
        'Return this exact JSON: {"title":"...","slides":[{"title":"...","body":"...","narration":"..."}]}\n'
        "narration should be 2-3 natural spoken sentences per slide.\n\n"
        f"Document:\n{text[:6000]}"
    )

    try:
        raw = ask_llm(provider, system, [{"role": "user", "content": user_msg}])
    except Exception as e:
        _gen_status[doc_id] = {"status": "error", "detail": str(e)}
        return

    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    try:
        script = json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                script = json.loads(m.group())
            except Exception:
                _gen_status[doc_id] = {"status": "error", "detail": "LLM returned invalid JSON. Try Groq or Gemini provider."}
                return
        else:
            _gen_status[doc_id] = {"status": "error", "detail": "LLM did not return JSON. Try a different provider."}
            return

    slides = script.get("slides", [])
    if not slides:
        _gen_status[doc_id] = {"status": "error", "detail": "No slides generated."}
        return

    output_path = str(STORAGE / f"{doc_id}.generated.mp4")
    try:
        build_video(slides, output_path, lang=lang)
        _gen_status[doc_id] = {"status": "ready", "title": script.get("title", "Presentation")}
    except Exception as e:
        _gen_status[doc_id] = {"status": "error", "detail": f"Video build failed: {e}"}


@app.post("/api/generate-video")
def generate_video_endpoint(req: VideoGenRequest, background_tasks: BackgroundTasks):
    text = _get_doc(req.doc_id)
    _gen_status[req.doc_id] = {"status": "generating"}
    background_tasks.add_task(_do_generate_video, req.doc_id, req.provider, text, req.lang)
    return {"status": "generating"}


@app.get("/api/video-status/{doc_id}")
def video_status(doc_id: str):
    return _gen_status.get(doc_id, {"status": "idle"})


@app.get("/api/download-video/{doc_id}")
def download_video(doc_id: str):
    path = STORAGE / f"{doc_id}.generated.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No generated video found.")
    meta = _load_meta(doc_id)
    base = meta.get("name", doc_id).rsplit(".", 1)[0] if meta else doc_id
    return FileResponse(str(path), media_type="video/mp4", filename=f"{base}_presentation.mp4")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_doc(doc_id: str) -> str:
    text = _load_text(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="Document not found. Please upload it again.")
    return text


app.mount("/", StaticFiles(directory="static", html=True), name="static")
