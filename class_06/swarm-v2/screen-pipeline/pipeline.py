"""
screen-pipeline/pipeline.py
─────────────────────────────
The core pipeline the swarm is training.

Capture → Describe → Store in Hindsight → Query by similarity

All inference is local via Ollama. Memory is managed by Hindsight
(auto-embeds, auto-indexes — no manual vector management needed).

Usage:
    python pipeline.py --mode demo      # no screen, synthetic frames
    python pipeline.py --mode live      # real screen capture
    python pipeline.py --search "debugging"  # find similar frames
"""

import argparse
import base64
import os
import sys
from io import BytesIO

from dotenv import load_dotenv

# Add parent dir so we can import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import ollama
from shared import hindsight

load_dotenv()

DEMO_FRAMES = [
    "Excel spreadsheet with quarterly revenue data. Multiple tabs open. Formulas visible.",
    "Figma canvas. Landing page wireframe being refined. Designer dragging components.",
    "Browser. Google Scholar. Searching research articles on machine learning.",
    "Slack. Team channel. Reading messages. Not typing.",
    "Google Docs. Two proposal drafts side by side. Comparing versions.",
]

# ─── Step 1: Capture ─────────────────────────────────────────────

def capture_screen() -> tuple[str, str | None]:
    """Returns (raw_description, base64_image_or_None)"""
    try:
        import pyautogui
        ss  = pyautogui.screenshot()
        buf = BytesIO()
        ss.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return "live frame", b64
    except Exception:
        return DEMO_FRAMES[0], None


# ─── Step 2: Describe (this is what the swarm improves) ──────────

def describe_frame(
    raw_description: str,
    image_b64: str | None,
    approach: str | None = None,
) -> str:
    """
    Convert a screen frame into text suitable for embedding.

    `approach` is what the swarm's refiner proposes.
    When no approach is set, we use the default.
    The swarm's job is to make this produce better embeddings.
    """

    default_approach = (
        "Describe what the person is doing, their cognitive state, "
        "intent, work stage, and the tool context. "
        "Focus on MEANING over surface appearance. Max 150 words."
    )

    instructions = approach or default_approach

    prompt = f"Screen frame: \"{raw_description}\"\n\n{instructions}"

    return ollama.chat(
        prompt=prompt,
        max_tokens=200,
        image_b64=image_b64,
    )


# ─── Step 3: Store in Hindsight ──────────────────────────────────

def store_frame(
    description: str,
    approach: str,
    agent: str,
    score: float | None = None,
) -> dict:
    """
    Store frame description in Hindsight.
    Hindsight auto-embeds and indexes — no manual vector management.
    """
    context = f"approach={approach} | agent={agent}"
    if score is not None:
        context += f" | score={score}"
    return hindsight.retain(content=description, context=context)


# ─── Step 4: Query by similarity ─────────────────────────────────

def find_similar(query_text: str, limit: int = 5) -> list[dict]:
    """
    Find stored frames semantically similar to a query.
    Hindsight handles embedding + search internally.
    """
    return hindsight.recall(query=query_text, limit=limit)


# ─── Demo runner ──────────────────────────────────────────────────

def run_demo(mode: str, search_query: str | None):
    print(f"\n{'='*60}")
    print(f"  Screen Pipeline — mode: {mode}")
    print(f"  Model: {ollama.OLLAMA_MODEL} (local via Ollama)")
    print(f"  Memory: Hindsight at {hindsight.HINDSIGHT_URL}")
    print(f"{'='*60}\n")

    hindsight_ok = hindsight.is_available()
    if not hindsight_ok:
        print("⚠️  Hindsight is not running — start with: docker compose up -d")
        print("   Running without storage.\n")

    if search_query and hindsight_ok:
        print(f"Searching for: '{search_query}'\n")
        results = find_similar(search_query)
        for r in results:
            content = r.get("content", "")[:80]
            print(f"  {content}")
        return

    # Process frames
    frames = DEMO_FRAMES if mode == "demo" else [capture_screen()]

    for i, frame_input in enumerate(frames[:3], 1):
        if isinstance(frame_input, tuple):
            raw, img = frame_input
        else:
            raw, img = frame_input, None

        print(f"── Frame {i} ───────────────────────────────────")
        print(f"Raw: {raw[:80]}")

        description = describe_frame(raw, img)
        print(f"Described: {description[:150]}...")

        if hindsight_ok:
            result = store_frame(
                description = description,
                approach    = "default",
                agent       = "pipeline-demo",
                score       = None,
            )
            print(f"Stored in Hindsight: {result}")

        print()


def main():
    if not ollama.is_available():
        print("❌  Ollama is not running. Start it with: ollama serve"); return

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   default="demo", choices=["demo", "live"])
    parser.add_argument("--search", default=None,   help="Search query for similarity")
    args = parser.parse_args()

    run_demo(args.mode, args.search)


if __name__ == "__main__":
    main()
