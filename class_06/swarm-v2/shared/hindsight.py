"""
shared/hindsight.py
────────────────────
Shared Hindsight client for the screen pipeline.

Hindsight is an agentic memory system that handles embedding,
storage, and semantic retrieval internally. No manual vector
management needed — just retain() and recall().

Runs as a Docker container alongside Redis.
"""

import os
import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_URL", "http://localhost:8888")
BANK_ID       = os.getenv("HINDSIGHT_BANK", "screen-swarm")


def retain(
    content: str,
    context: str = "",
    bank_id: str | None = None,
) -> dict:
    """
    Store a memory in Hindsight.
    Content is automatically embedded and organized.
    """
    bank = bank_id or BANK_ID
    response = requests.post(
        f"{HINDSIGHT_URL}/v1/banks/{bank}/memories/retain",
        json={
            "content": content,
            "context": context,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def recall(
    query: str,
    limit: int = 5,
    bank_id: str | None = None,
) -> list[dict]:
    """
    Retrieve relevant memories from Hindsight.
    Uses semantic + keyword + graph search internally.
    """
    bank = bank_id or BANK_ID
    response = requests.post(
        f"{HINDSIGHT_URL}/v1/banks/{bank}/memories/recall",
        json={
            "query": query,
            "limit": limit,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("memories", [])


def is_available() -> bool:
    """Check if Hindsight is running."""
    try:
        r = requests.get(f"{HINDSIGHT_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False
