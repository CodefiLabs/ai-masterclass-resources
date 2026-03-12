"""
shared/ollama.py
────────────────
Shared Ollama client for all swarm agents.

All inference is local — no cloud API calls, no data leaves the machine.
Uses Ollama's REST API at localhost:11434.

Models:
  - qwen3.5:latest     → reasoning (agent prompts, scoring, refining)
  - qwen3-embedding    → text embeddings (if needed directly)
"""

import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")


def chat(
    prompt: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 1024,
    image_b64: str | None = None,
) -> str:
    """
    Send a chat completion request to Ollama.
    Returns the assistant's response text.

    Supports optional base64 image for vision models.
    """
    model = model or OLLAMA_MODEL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    user_content = prompt
    user_msg = {"role": "user", "content": user_content}

    # Vision model support: attach image if provided
    if image_b64:
        user_msg["images"] = [image_b64]

    messages.append(user_msg)

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":    model,
            "messages": messages,
            "stream":   False,
            "think":    False,   # disable thinking mode on qwen3.x/deepseek-r1 models
            "options":  {"num_predict": max_tokens},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def embed(text: str, model: str = "qwen3-embedding") -> list[float]:
    """
    Generate an embedding vector using Ollama's embedding API.
    Returns a list of floats.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def is_available() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def available_models() -> list[str]:
    """Return list of model names available in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
