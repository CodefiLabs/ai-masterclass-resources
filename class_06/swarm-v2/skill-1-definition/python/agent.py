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
    agent:        str
    what:         str   # What are you doing right now?
    goal:         str   # What's your goal?
    why:          str   # Why are you doing it?
    how:          str   # How are you approaching it?
    score:        float # Last known score (0.0 - 1.0)
    # Reinforcement fields — populated after each scored task
    what_worked:  str = ""   # what's working + strongest parts + what to keep
    what_didnt:   str = ""   # what's not working + weakest parts + what to change
    what_missing: str = ""   # what should have been included but wasn't
    help_needed:  str = ""
    timestamp:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MemoryEntry:
    task:         str
    result:       str
    score:        float
    what_worked:  str
    what_didnt:   str
    what_missing: str


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
        self.current_task = "idle"
        self.reasoning    = ""
        self.approach     = ""
        self.last_score   = 0.0
        self.what_worked  = ""
        self.what_didnt   = ""
        self.what_missing = ""

        # Rolling memory — last N scored results
        self.memory: list[MemoryEntry] = []
        self.memory_limit = 5

    # ── Six Questions ─────────────────────────────────────────────

    def get_status(self) -> AgentStatus:
        """Answer all six questions right now."""
        return AgentStatus(
            agent        = self.name,
            what         = self.current_task,
            goal         = self.goal,
            why          = self.reasoning,
            how          = self.approach,
            score        = self.last_score,
            what_worked  = self.what_worked,
            what_didnt   = self.what_didnt,
            what_missing = self.what_missing,
            help_needed  = self._assess_help(),
        )

    def _assess_help(self) -> str:
        """Question 6: How can I help?"""
        if self.last_score == 0.0:
            return "No score yet — just getting started"
        if self.last_score < 0.4:
            return f"Struggling (score {self.last_score:.2f}) — need a different approach for: {self.what_didnt}"
        if self.last_score < 0.7:
            return f"Making progress (score {self.last_score:.2f}) — could improve: {self.what_missing}"
        return f"Performing well (score {self.last_score:.2f}) — no help needed"

    # ── Memory ────────────────────────────────────────────────────

    def add_to_memory(self, task: str, result: str, score: dict):
        """Store a scored result. Updates live state for next question answers."""
        entry = MemoryEntry(
            task         = task,
            result       = result[:400],
            score        = score.get("overall", 0.0),
            what_worked  = score.get("what_worked", ""),
            what_didnt   = score.get("what_didnt", ""),
            what_missing = score.get("what_missing", ""),
        )
        self.memory.append(entry)
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)

        # Update live state so get_status() reflects latest
        self.last_score   = entry.score
        self.what_worked  = entry.what_worked
        self.what_didnt   = entry.what_didnt
        self.what_missing = entry.what_missing

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
            f"Worked: {m.what_worked} | "
            f"Didn't: {m.what_didnt} | "
            f"Missing: {m.what_missing}"
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

    # Simulate three rounds with increasing scores
    rounds = [
        {
            "task": "Describe this screen frame for embedding: Excel spreadsheet open with multiple tabs and a PDF in split screen",
            "score": {
                "overall": 0.45,
                "what_worked": "Identified the tools correctly; concise",
                "what_didnt":  "Too surface-level — missed cognitive state, just listed what's visible",
                "what_missing":"No mention of what stage of work this represents",
            }
        },
        {
            "task": "Describe this screen frame for embedding: Figma canvas with a landing page design being iterated on",
            "score": {
                "overall": 0.68,
                "what_worked": "Captured the creative design context; included tool and task",
                "what_didnt":  "Didn't infer enough about the person's intent behind the iteration",
                "what_missing":"Should note this is a focused creative/refinement state",
            }
        },
        {
            "task": "Describe this screen frame for embedding: Slack open with browser showing research articles in background",
            "score": {
                "overall": 0.82,
                "what_worked": "Inferred context switch and research mode; captured cognitive state well",
                "what_didnt":  "Could note the distraction pattern more explicitly — person may have been pulled off-task",
                "what_missing":"",
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

        status = agent.get_status()
        print(f"\nSix Questions after round {i}:")
        print(f"  Score:        {status.score:.2f}")
        print(f"  What worked:  {status.what_worked}")
        print(f"  What didn't:  {status.what_didnt}")
        print(f"  Help needed:  {status.help_needed}")
        print()
