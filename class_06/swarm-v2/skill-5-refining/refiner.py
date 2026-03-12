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
        """Analyze N scores worth of impact items, find patterns, propose a better approach.
        Also proposes adjustments to capture parameters (interval, batch window, max frames)."""
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

        # ── Step 1: Propose refined embedding approach ──
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

        # ── Step 2: Propose capture parameter adjustments ──
        await self._adjust_capture_config(avg_score, all_items, history_summary)

    async def _adjust_capture_config(self, avg_score: float, all_items: list, history_summary: str):
        """Use LLM to propose adjustments to capture parameters based on scoring patterns."""
        # Read current config
        current_config = {"capture_interval": 2, "batch_window": 30, "max_frames_per_batch": 15}
        try:
            raw = await self.redis.get("swarm:capture_config")
            if raw:
                current_config = json.loads(raw)
        except Exception:
            pass

        config_response = ollama.chat(
            prompt=(
                f"Current capture configuration:\n"
                f"  capture_interval: {current_config.get('capture_interval', 2)} seconds (range: 1-5)\n"
                f"  batch_window: {current_config.get('batch_window', 30)} seconds (range: 15-60)\n"
                f"  max_frames_per_batch: {current_config.get('max_frames_per_batch', 15)} (range: 5-30)\n\n"
                f"Average score from last batch: {avg_score:.2f}\n"
                f"Total impact items: {len(all_items)}\n\n"
                f"Scoring pattern summary:\n{history_summary}\n\n"
                f"Based on the scoring patterns, recommend adjustments to capture parameters.\n\n"
                f"Trade-offs to consider:\n"
                f"- More frequent captures (lower interval) = more data but more noise and processing load\n"
                f"- Less frequent captures (higher interval) = cleaner frames but might miss transitions\n"
                f"- Larger batch windows = more context for understanding activity arcs, but slower feedback loop\n"
                f"- Smaller batch windows = faster iteration but less context per analysis\n"
                f"- More frames per batch = richer picture but longer LLM processing time\n"
                f"- Fewer frames per batch = faster processing but might miss key moments\n\n"
                f"Return ONLY valid JSON with your recommended values:\n"
                f'{{"capture_interval": <1-5>, "batch_window": <15-60>, "max_frames_per_batch": <5-30>}}'
            ),
            system=(
                "You are a systems tuner for a screen capture pipeline. "
                "You adjust capture parameters based on scoring evidence. "
                "If scores are high (>0.7), make conservative changes. "
                "If scores are low (<0.5), try more aggressive adjustments. "
                "If specificity scores are low, consider more frequent captures. "
                "If noise_resistance scores are low, consider less frequent captures. "
                "Return ONLY the JSON object, no other text."
            ),
            max_tokens=128,
        )

        try:
            clean = config_response.strip().strip("```json").strip("```").strip()
            new_config = json.loads(clean)
            # Clamp values to valid ranges
            new_config["capture_interval"] = max(1, min(5, int(new_config.get("capture_interval", 2))))
            new_config["batch_window"] = max(15, min(60, int(new_config.get("batch_window", 30))))
            new_config["max_frames_per_batch"] = max(5, min(30, int(new_config.get("max_frames_per_batch", 15))))

            await self.redis.set("swarm:capture_config", json.dumps(new_config))

            print(f"[refiner] Capture config updated: interval={new_config['capture_interval']}s, "
                  f"window={new_config['batch_window']}s, max_frames={new_config['max_frames_per_batch']}")
        except Exception as e:
            print(f"[refiner] Failed to parse capture config response, keeping current: {e}")
            print(f"  Raw response: {config_response[:200]}")


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
        async def get(self, *a, **k): return None
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
