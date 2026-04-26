import os
import httpx


PROVIDERS = {
    "llamafile": {
        "label": "Qwen3.5 0.8B (Local — Free)",
        "base_url": "http://localhost:8080/v1/chat/completions",
        "model": "Qwen3.5-0.8B",
        "key_env": "",  # no key needed
    },
    "groq": {
        "label": "Groq (Llama 3.3 70B) — Free",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "label": "OpenRouter (Mistral 7B) — Free",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "mistralai/mistral-7b-instruct:free",
        "key_env": "OPENROUTER_API_KEY",
    },
    "nvidia_nim": {
        "label": "NVIDIA NIM (Llama 4 Maverick) — Free",
        "base_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "key_env": "NVIDIA_NIM_API_KEY",
    },
    "gemini": {
        "label": "Google Gemini 2.0 Flash Lite — Free",
        "base_url": None,
        "model": "gemini-2.0-flash-lite",
        "key_env": "GEMINI_API_KEY",
    },
}


def get_available_providers() -> list[dict]:
    return [
        {
            "id": key,
            "label": cfg["label"],
            "configured": True if not cfg["key_env"] else bool(os.environ.get(cfg["key_env"], "")),
        }
        for key, cfg in PROVIDERS.items()
    ]


def ask_llm(provider_id: str, system_prompt: str, messages: list[dict]) -> str:
    """
    messages: list of {"role": "user"/"assistant", "content": "..."}
    Includes full conversation history + the latest user message.
    """
    cfg = PROVIDERS.get(provider_id)
    if not cfg:
        raise ValueError(f"Unknown provider: {provider_id}")

    api_key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else "no-key"
    if cfg["key_env"] and not api_key:
        raise ValueError(f"API key not set for {provider_id} — set {cfg['key_env']} in your .env")

    if provider_id == "gemini":
        return _ask_gemini(cfg["model"], api_key, system_prompt, messages)

    return _ask_openai_compat(cfg["base_url"], cfg["model"], api_key, system_prompt, messages)


def _ask_openai_compat(
    base_url: str, model: str, api_key: str, system: str, messages: list[dict]
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 2048,
        "temperature": 1.00,
        "top_p": 1.00,
        "frequency_penalty": 0.00,
        "presence_penalty": 0.00,
    }
    resp = httpx.post(base_url, json=payload, headers=headers, timeout=60)
    if not resp.is_success:
        raise ValueError(f"API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def _ask_gemini(model: str, api_key: str, system: str, messages: list[dict]) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model, system_instruction=system)

    history = []
    for msg in messages[:-1]:
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        })

    chat = m.start_chat(history=history)
    response = chat.send_message(messages[-1]["content"])
    return response.text


def analyze_video_with_openrouter(frames: list[dict], prompt: str) -> str:
    """
    Send video frames to OpenRouter's free Llama 3.2 Vision model.
    Uses at most 4 frames to stay within free-tier payload limits.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set — required for OpenRouter vision")

    sample = frames[:4]  # keep payload manageable on free tier
    content: list[dict] = []
    for f in sample:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['b64']}"}})
        content.append({"type": "text", "text": f"Frame at {f['time']}"})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
        "temperature": 0.4,
    }
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=90,
    )
    if not resp.is_success:
        raise ValueError(f"OpenRouter vision error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def analyze_video_with_gemini(frames: list[dict], prompt: str) -> str:
    """
    frames: list of {"b64": str, "time": "M:SS"}
    Sends frames as inline images to Gemini 2.0 Flash for video analysis.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set — required for video analysis")

    parts: list[dict] = []
    for f in frames:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": f["b64"]}})
        parts.append({"text": f"Frame at {f['time']}"})
    parts.append({"text": prompt})

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {
            "parts": [{"text": (
                "You are a professional video analyst. Analyze the provided video frames "
                "and give a detailed, structured response about the video content."
            )}]
        },
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.4},
    }

    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
        json=body,
        timeout=60,
    )
    if not resp.is_success:
        raise ValueError(f"Gemini vision error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
