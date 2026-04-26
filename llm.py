import os
import httpx


PROVIDERS = {
    "groq": {
        "label": "Groq (Llama 3.3 70B) — Free",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "label": "OpenRouter (Llama 3.1 8B) — Free",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "key_env": "OPENROUTER_API_KEY",
    },
    "nvidia_nim": {
        "label": "NVIDIA NIM (Llama 4 Maverick) — Free",
        "base_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "key_env": "NVIDIA_NIM_API_KEY",
    },
    "gemini": {
        "label": "Google Gemini 1.5 Flash — Free",
        "base_url": None,
        "model": "gemini-1.5-flash",
        "key_env": "GEMINI_API_KEY",
    },
}


def get_available_providers() -> list[dict]:
    return [
        {
            "id": key,
            "label": cfg["label"],
            "configured": bool(os.environ.get(cfg["key_env"], "")),
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

    api_key = os.environ.get(cfg["key_env"], "")
    if not api_key:
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
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _ask_gemini(model: str, api_key: str, system: str, messages: list[dict]) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model, system_instruction=system)

    # Convert history to Gemini format
    history = []
    for msg in messages[:-1]:
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        })

    chat = m.start_chat(history=history)
    response = chat.send_message(messages[-1]["content"])
    return response.text
