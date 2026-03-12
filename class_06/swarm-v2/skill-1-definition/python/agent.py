"""
skill-1-definition/python/agent.py
───────────────────────────────────
SKILL 1: Agent Definition

Run this to see an agent answer the six questions,
then improve its next response based on a simulated score.

All inference is local via Ollama — no cloud API calls.

Usage:
    pip install -r requirements.txt
    python agent.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Add shared/ to path for ollama helper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
import ollama

load_dotenv()


# ─── The Six Questions ────────────────────────────────────────────
# Every agent must answer these at any time.
# They aren't just logging — they're the collaboration protocol.
# Other agents read these to decide how to help.

@dataclass
class AgentStatus:
    agent:         str
    what:          str   # What are you doing right now?
    goal:          str   # What's your goal?
    why:           str   # Why are you doing it?
    how:           str   # How are you approaching it?
    score:         float # Last known score (0.0 - 1.0)
    # Reinforcement fields — populated after each scored task
    top_positive:  str = ""   # highest-impact positive observation
    top_negative:  str = ""   # highest-impact negative observation
    help_needed:   str = ""
    timestamp:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MemoryEntry:
    task:          str
    result:        str
    score:         float
    top_positive:  str
    top_negative:  str
    impact_items:  list = field(default_factory=list)


# ─── Two-Pass Scoring ────────────────────────────────────────────
# Pass 1 (LLM): itemize observations with +/- impact scores
# Pass 2 (math): net_impact / √(items) → normalized 0-1 score

def derive_score(items: list[dict]) -> dict:
    """Pass 2: deterministic math. No LLM involved."""
    if not items:
        return {"overall": 0.5, "net_impact": 0, "total_items": 0,
                "normalized_impact": 0.0, "raw_score": 50.0}
    net_impact = sum(item["impact"] for item in items)
    total_items = len(items)
    normalized_impact = net_impact / math.sqrt(total_items)
    raw_score = max(0, min(100, 50 + (normalized_impact * 8.0)))
    overall = round(raw_score / 100, 3)
    return {"overall": overall, "net_impact": net_impact, "total_items": total_items,
            "normalized_impact": round(normalized_impact, 3), "raw_score": round(raw_score, 1)}


class Agent:
    """
    Base agent class.
    Subclass this and implement `act()`.
    Everything else (six questions, memory, prompt building) is inherited.

    All LLM calls go through Ollama on localhost — fully local.
    """

    def __init__(self, name: str, role: str, goal: str):
        self.name   = name
        self.role   = role
        self.goal   = goal

        # Live state (answers to six questions)
        self.current_task  = "idle"
        self.reasoning     = ""
        self.approach      = ""
        self.last_score    = 0.0
        self.top_positive  = ""
        self.top_negative  = ""

        # Rolling memory — last N scored results
        self.memory: list[MemoryEntry] = []
        self.memory_limit = 5

    # ── Six Questions ─────────────────────────────────────────────

    def get_status(self) -> AgentStatus:
        """Answer all six questions right now."""
        return AgentStatus(
            agent         = self.name,
            what          = self.current_task,
            goal          = self.goal,
            why           = self.reasoning,
            how           = self.approach,
            score         = self.last_score,
            top_positive  = self.top_positive,
            top_negative  = self.top_negative,
            help_needed   = self._assess_help(),
        )

    def _assess_help(self) -> str:
        """Question 6: How can I help?"""
        if self.last_score == 0.0:
            return "No score yet — just getting started"
        if self.last_score < 0.4:
            return f"Struggling (score {self.last_score:.2f}) — top issue: {self.top_negative}"
        if self.last_score < 0.7:
            return f"Making progress (score {self.last_score:.2f}) — working on: {self.top_negative}"
        return f"Performing well (score {self.last_score:.2f}) — no help needed"

    # ── Memory ────────────────────────────────────────────────────

    def add_to_memory(self, task: str, result: str, score: dict):
        """Store a scored result. Updates live state for next question answers."""
        items = score.get("items", [])
        derived = score.get("derived", derive_score(items))

        # Extract top positive/negative from impact items
        positives = sorted([i for i in items if i["impact"] > 0],
                           key=lambda x: x["impact"], reverse=True)
        negatives = sorted([i for i in items if i["impact"] < 0],
                           key=lambda x: x["impact"])
        top_pos = positives[0]["observation"] if positives else ""
        top_neg = negatives[0]["observation"] if negatives else ""

        entry = MemoryEntry(
            task          = task,
            result        = result[:400],
            score         = derived.get("overall", 0.5),
            top_positive  = top_pos,
            top_negative  = top_neg,
            impact_items  = items,
        )
        self.memory.append(entry)
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)

        # Update live state so get_status() reflects latest
        self.last_score   = entry.score
        self.top_positive = entry.top_positive
        self.top_negative = entry.top_negative

    def build_system_prompt(self) -> str:
        """
        The system prompt is where memory becomes learning.
        Recent scores are injected so the agent adjusts its approach
        without being retrained — just re-prompted.
        """
        base = (
            f"You are {self.name}, a {self.role}.\n"
            f"Goal: {self.goal}\n\n"
            f"After completing any task, structure your response as:\n"
            f"WHAT: [what you did]\n"
            f"HOW: [how you approached it]\n"
            f"RESULT: [the output]\n"
            f"CONFIDENCE: [0.0-1.0]\n"
            f"HELP: [what would make your next attempt better]"
        )

        if not self.memory:
            return base

        history = "\n\n".join([
            f"Task: {m.task[:80]}\n"
            f"Score: {m.score:.2f} | "
            f"Best: {m.top_positive} | "
            f"Worst: {m.top_negative}"
            for m in self.memory[-3:]
        ])

        return (
            f"{base}\n\n"
            f"Your recent performance history — use this to improve:\n"
            f"{history}"
        )

    # ── LLM (Ollama — fully local) ───────────────────────────────

    def ask(self, prompt: str, image_b64: str | None = None) -> str:
        """Send a prompt to the local Ollama model."""
        return ollama.chat(
            prompt    = prompt,
            system    = self.build_system_prompt(),
            image_b64 = image_b64,
        )

    def act(self, task: str) -> str:
        """Override in subclasses. Default: just ask the local model."""
        self.current_task = task
        return self.ask(task)


# ─── Demo: watch an agent learn over 3 rounds ────────────────────

if __name__ == "__main__":
    if not ollama.is_available():
        print("❌  Ollama is not running. Start it with: ollama serve")
        print(f"   Expected at: {ollama.OLLAMA_BASE_URL}")
        exit(1)

    agent = Agent(
        name = "embedder",
        role = "Screen Embedding Strategist",
        goal = "Produce embedding descriptions that best capture human work intent from screen frames",
    )

    # Simulate three rounds with increasing scores (two-pass format)
    # Pass 1: itemized impact observations  |  Pass 2: derive_score() does the math
    rounds = [
        {
            "task": "Describe this screen frame for embedding: Excel spreadsheet open with multiple tabs and a PDF in split screen",
            "score": {
                "items": [
                    {"dimension": "intent_capture",   "observation": "Identified tools correctly",        "impact": +1},
                    {"dimension": "specificity",      "observation": "Concise output",                    "impact": +1},
                    {"dimension": "intent_capture",   "observation": "Missed cognitive state entirely",   "impact": -3},
                    {"dimension": "specificity",      "observation": "Just listed what's visible",        "impact": -2},
                    {"dimension": "noise_resistance", "observation": "No mention of work stage",          "impact": -2},
                ],
                # derive_score() → overall ≈ 0.46 (net=-5, √5=2.24, norm=-2.24, raw=32.1)
            }
        },
        {
            "task": "Describe this screen frame for embedding: Figma canvas with a landing page design being iterated on",
            "score": {
                "items": [
                    {"dimension": "intent_capture",    "observation": "Captured creative design context",  "impact": +3},
                    {"dimension": "specificity",       "observation": "Included tool and task correctly",  "impact": +2},
                    {"dimension": "cognitive_state",   "observation": "Noted iteration pattern",           "impact": +1},
                    {"dimension": "intent_capture",    "observation": "Didn't infer intent behind iteration", "impact": -2},
                    {"dimension": "cognitive_state",   "observation": "Should note focused creative state", "impact": -1},
                ],
                # derive_score() → overall ≈ 0.61 (net=+3, √5=2.24, norm=1.34, raw=60.7)
            }
        },
        {
            "task": "Describe this screen frame for embedding: Slack open with browser showing research articles in background",
            "score": {
                "items": [
                    {"dimension": "intent_capture",    "observation": "Inferred context switch accurately", "impact": +3},
                    {"dimension": "cognitive_state",   "observation": "Identified research mode",           "impact": +3},
                    {"dimension": "specificity",       "observation": "Distinguished Slack from browser",   "impact": +2},
                    {"dimension": "noise_resistance",  "observation": "Focused on work-relevant signals",   "impact": +2},
                    {"dimension": "cognitive_state",   "observation": "Could note distraction pattern",     "impact": -1},
                ],
                # derive_score() → overall ≈ 0.82 (net=+9, √5=2.24, norm=4.02, raw=82.2)
            }
        },
    ]

    print(f"\n{'═'*60}")
    print(f"  SKILL 1: Agent Definition Demo")
    print(f"  Agent: {agent.name} | Role: {agent.role}")
    print(f"  Model: {ollama.OLLAMA_MODEL} (local via Ollama)")
    print(f"{'═'*60}\n")

    for i, round_data in enumerate(rounds, 1):
        print(f"── Round {i} ──────────────────────────────────")
        print(f"Task: {round_data['task'][:80]}...")
        print(f"\nSystem prompt preview (first 200 chars):")
        print(f"  {agent.build_system_prompt()[:200]}...")

        result = agent.act(round_data["task"])
        print(f"\nAgent response (first 300 chars):")
        print(f"  {result[:300]}...")

        agent.add_to_memory(round_data["task"], result, round_data["score"])

        # Show the two-pass scoring
        items = round_data["score"]["items"]
        derived = derive_score(items)
        print(f"\nTwo-pass scoring:")
        print(f"  Pass 1 — {len(items)} impact items:")
        for item in items:
            sign = "+" if item["impact"] > 0 else ""
            print(f"    [{item['dimension']}] {sign}{item['impact']}: {item['observation']}")
        print(f"  Pass 2 — derive_score():")
        print(f"    net_impact={derived['net_impact']}  √items={math.sqrt(len(items)):.2f}  overall={derived['overall']}")

        status = agent.get_status()
        print(f"\nSix Questions after round {i}:")
        print(f"  Score:         {status.score:.2f}")
        print(f"  Top positive:  {status.top_positive}")
        print(f"  Top negative:  {status.top_negative}")
        print(f"  Help needed:   {status.help_needed}")
        print()
