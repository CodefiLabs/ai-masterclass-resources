"""
skill-5-refining/refiner.py
─────────────────────────────
SKILL 5: Agent Refining

All inference is local via Ollama — no cloud API calls.

The refiner closes the loop. It:
  1. Reads score history from screen:results (impact items + derived scores)
  2. Groups items by dimension, finds recurring patterns
  3. Proposes a new approach weighted by impact evidence
  4. Publishes the new approach back to the embedder agents

The refiner runs on a slower cadence (every N scores, not every frame).

Usage:
    python refiner.py              # watch screen:results, refine every 5 scores
    python refiner.py --simulate   # run on fake history to see output
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import redis.asyncio as aioredis
from dotenv import load_dotenv

# Add parent dir so we can import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import ollama

load_dotenv()

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379")
REFINE_EVERY = 5   # refine after N scored results


# ─── Refiner Agent ────────────────────────────────────────────────

class RefinerAgent:
    """
    Reads score history with impact items.
    Finds patterns in what worked and what didn't across dimensions.
    Proposes a refined approach weighted by impact evidence.
    Writes it back to a Redis key that embedder agents watch.
    """

    def __init__(self):
        self.redis  = None
        self.score_buffer: list[dict] = []

    async def start(self):
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        print("[refiner] Connected to Redis")

    async def run(self):
        await self.start()
        try:
            await self.redis.xgroup_create("screen:results", "refiners", id="0", mkstream=True)
        except Exception:
            pass

        print(f"[refiner] Watching screen:results — will refine every {REFINE_EVERY} scores")
        print(f"[refiner] Model: {ollama.OLLAMA_MODEL} (local via Ollama)")

        while True:
            results = await self.redis.xreadgroup(
                groupname="refiners", consumername="refiner-main",
                streams={"screen:results": ">"}, count=1, block=2000,
            )
            if results:
                for _, messages in results:
                    for msg_id, data in messages:
                        msg = json.loads(data["data"])
                        payload = msg.get("payload", {})
                        self.score_buffer.append(payload)
                        score_val = payload.get("score", {}).get("overall", 0)
                        item_count = payload.get("score", {}).get("total_items", 0)
                        await self.redis.xack("screen:results", "refiners", msg_id)
                        print(f"  [refiner] Buffered score #{len(self.score_buffer)}: "
                              f"{score_val:.2f} ({item_count} items)")

                        if len(self.score_buffer) >= REFINE_EVERY:
                            await self._refine()
                            self.score_buffer = []

    async def _refine(self):
        """Analyze N scores worth of impact items, find patterns, propose a better approach."""
        print(f"\n[refiner] Running refinement on {len(self.score_buffer)} scores...")

        # Collect all impact items across buffered scores
        all_items = []
        for payload in self.score_buffer:
            all_items.extend(payload.get("items", []))

        # Calculate average overall score
        scores = [p.get("score", {}).get("overall", 0) for p in self.score_buffer]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Group items by dimension, separate positives and negatives
        by_dimension = {}
        for item in all_items:
            dim = item.get("dimension", "unknown")
            by_dimension.setdefault(dim, []).append(item)

        # Build evidence-weighted history summary
        summary_parts = [
            f"Average score over last {len(self.score_buffer)} attempts: {avg_score:.2f}",
            f"Total impact items analyzed: {len(all_items)}",
            "",
        ]

        for dim in ["intent_capture", "cognitive_state", "specificity", "noise_resistance"]:
            items = by_dimension.get(dim, [])
            if not items:
                summary_parts.append(f"{dim.upper()}: No observations")
                continue

            positives = sorted([i for i in items if i["impact"] > 0], key=lambda x: -x["impact"])
            negatives = sorted([i for i in items if i["impact"] < 0], key=lambda x: x["impact"])

            summary_parts.append(f"\n{dim.upper()}:")
            if positives:
                summary_parts.append("  Strengths (keep doing):")
                for p in positives[:3]:
                    summary_parts.append(f"    (+{p['impact']}) {p['observation']}")
            if negatives:
                summary_parts.append("  Weaknesses (change or drop):")
                for n in negatives[:3]:
                    summary_parts.append(f"    ({n['impact']}) {n['observation']}")

        history_summary = "\n".join(summary_parts)

        refined = ollama.chat(
            prompt=(
                f"Here is the impact analysis from the last {len(self.score_buffer)} embedding attempts:\n\n"
                f"{history_summary}\n\n"
                f"Based on this evidence, propose a refined embedding approach.\n"
                f"The approach will be injected into every embedder agent's system prompt.\n\n"
                f"APPROACH_NAME: [short name for this approach]\n"
                f"INSTRUCTIONS: [specific instructions for the embedder — what to include, what to avoid]\n"
                f"HYPOTHESIS: [why you think this will score higher]\n"
                f"SUCCESS_SIGNAL: [what a score of 0.8+ would look like for this approach]"
            ),
            system=(
                "You are a research director for an AI swarm that is learning to "
                "understand human work behavior from screen frames.\n"
                "You read impact-scored evidence and propose refined embedding strategies.\n"
                "Pay close attention to high-impact items (+3 and +5) — these are the strongest "
                "signals. Negative items (-3 and -5) are the most urgent problems to fix.\n"
                "Be specific and concrete. Give the agents something actionable to try."
            ),
            max_tokens=512,
        )

        await self.redis.set("swarm:current_approach", refined)
        await self.redis.xadd("agent:health", {
            "data": json.dumps({
                "type":        "approach_updated",
                "agent":       "refiner",
                "avg_score":   avg_score,
                "total_items": len(all_items),
                "new_approach": refined[:200],
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            })
        })

        print(f"[refiner] New approach written to Redis")
        print(f"  Avg score was: {avg_score:.2f}")
        print(f"  Items analyzed: {len(all_items)}")
        print(f"  Approach preview: {refined[:200]}...")


# ─── Read current approach (used by embedder agents) ─────────────

async def get_current_approach(redis_url: str = REDIS_URL) -> str | None:
    r = aioredis.from_url(redis_url, decode_responses=True)
    approach = await r.get("swarm:current_approach")
    await r.aclose()
    return approach


# ─── Simulation mode ──────────────────────────────────────────────

SIMULATED_SCORES = [
    {
        "score": {"overall": 0.45, "net_impact": -4, "total_items": 4},
        "items": [
            {"dimension": "intent_capture", "observation": "Identified the tool but not the task", "impact": 1},
            {"dimension": "cognitive_state", "observation": "No cognitive signal at all", "impact": -3},
            {"dimension": "specificity", "observation": "Too generic — 'working with numbers'", "impact": -2},
            {"dimension": "noise_resistance", "observation": "Listed tab count — irrelevant detail", "impact": -1},
        ],
    },
    {
        "score": {"overall": 0.52, "net_impact": -3, "total_items": 4},
        "items": [
            {"dimension": "intent_capture", "observation": "Named the activity as data work", "impact": 2},
            {"dimension": "cognitive_state", "observation": "No cognitive state mentioned", "impact": -2},
            {"dimension": "specificity", "observation": "Could apply to any spreadsheet task", "impact": -2},
            {"dimension": "noise_resistance", "observation": "Mentioned 'PDF visible' without purpose", "impact": -1},
        ],
    },
    {
        "score": {"overall": 0.48, "net_impact": -2, "total_items": 4},
        "items": [
            {"dimension": "intent_capture", "observation": "Said 'person working' — too vague", "impact": -2},
            {"dimension": "cognitive_state", "observation": "No cognitive signal", "impact": -2},
            {"dimension": "specificity", "observation": "Short and concise at least", "impact": 1},
            {"dimension": "noise_resistance", "observation": "Didn't describe window chrome", "impact": 1},
        ],
    },
    {
        "score": {"overall": 0.61, "net_impact": 1, "total_items": 4},
        "items": [
            {"dimension": "intent_capture", "observation": "Started mentioning analytical context", "impact": 2},
            {"dimension": "cognitive_state", "observation": "Still no cognitive state", "impact": -2},
            {"dimension": "specificity", "observation": "Referenced revenue specifically", "impact": 2},
            {"dimension": "noise_resistance", "observation": "Listed tools but not their purpose", "impact": -1},
        ],
    },
    {
        "score": {"overall": 0.55, "net_impact": 0, "total_items": 4},
        "items": [
            {"dimension": "intent_capture", "observation": "Mentioned cross-referencing as task", "impact": 2},
            {"dimension": "cognitive_state", "observation": "No focus/stuck/exploring signal", "impact": -2},
            {"dimension": "specificity", "observation": "Work stage mentioned but vague", "impact": 1},
            {"dimension": "noise_resistance", "observation": "Too descriptive not semantic", "impact": -1},
        ],
    },
]


async def simulate():
    agent = RefinerAgent()
    agent.score_buffer = SIMULATED_SCORES
    class MockRedis:
        async def set(self, *a, **k): pass
        async def xadd(self, *a, **k): pass
    agent.redis = MockRedis()
    await agent._refine()


def main():
    if not ollama.is_available():
        print("❌  Ollama is not running. Start it with: ollama serve"); return

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  SKILL 5: Agent Refining")
    print(f"  Model: {ollama.OLLAMA_MODEL} (local via Ollama)")
    print(f"{'='*60}\n")

    if args.simulate:
        print("Running simulation on fake impact history...\n")
        asyncio.run(simulate())
    else:
        r = RefinerAgent()
        asyncio.run(r.run())


if __name__ == "__main__":
    main()
