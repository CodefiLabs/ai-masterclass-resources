"""
skill-2-orchestration/run_swarm.py
────────────────────────────────────
SKILL 2: Agent Orchestration

Three agents wired together via Redis Streams.
Each does one job. They don't call each other directly —
they publish to streams and consume what they need.

All inference is local via Ollama — no cloud API calls.

AGENTS:
  FrameCaptureAgent   → captures screen, publishes to screen:frames
  EmbedderAgent       → reads screen:frames, proposes embedding approach, publishes to screen:approaches
  ApproachTestAgent   → reads screen:approaches, tests it, publishes result to screen:results

THE LOOP:
  capture → propose approach → score (two-pass) → feed back

Usage:
    pip install -r requirements.txt
    python run_swarm.py --mode demo      # synthetic frames, no screen
    python run_swarm.py --mode live      # real screen capture
    python run_swarm.py --agent embedder # run one agent only
"""

import asyncio
import argparse
import base64
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from io import BytesIO

import redis.asyncio as aioredis
from dotenv import load_dotenv

# Add parent dir so we can import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import ollama

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ─── Synthetic frames for demo mode ──────────────────────────────
DEMO_FRAMES = [
    "Excel open. Spreadsheet with revenue data. Multiple tabs. Formulas in cells.",
    "Figma canvas. Landing page wireframe. Designer dragging components.",
    "Browser. Google Scholar. Searching for research articles on machine learning.",
    "Slack. Team channel. Reading messages, not typing. Notifications visible.",
    "Google Docs. Two documents side by side. Comparing proposal drafts.",
    "Zendesk. Support ticket queue. Agent triaging incoming tickets.",
    "Notion. Project board. Moving cards between columns. Planning view.",
    "Browser. Chat window. Asking about data analysis approach.",
]


# ─── Two-Pass Scoring (shared with eval.py) ──────────────────────

DIMENSIONS = [
    "intent_capture",
    "cognitive_state",
    "specificity",
    "noise_resistance",
]


def derive_score(items: list[dict]) -> dict:
    """
    Pass 2: Derive normalized score from impact items.
    net_impact / sqrt(total_items) rewards quality over quantity.
    """
    if not items:
        return {
            "overall": 0.5,
            "net_impact": 0,
            "total_items": 0,
            "normalized_impact": 0.0,
            "raw_score": 50.0,
        }

    net_impact = sum(item["impact"] for item in items)
    total_items = len(items)
    normalized_impact = net_impact / math.sqrt(total_items)

    raw_score = max(0, min(100, 50 + (normalized_impact * 8.0)))
    overall = round(raw_score / 100, 3)

    return {
        "overall": overall,
        "net_impact": net_impact,
        "total_items": total_items,
        "normalized_impact": round(normalized_impact, 3),
        "raw_score": round(raw_score, 1),
    }


def dimension_scores(items: list[dict]) -> dict:
    """Derive per-dimension scores from impact items."""
    scores = {}
    for dim in DIMENSIONS:
        dim_items = [i for i in items if i.get("dimension") == dim]
        scores[dim] = derive_score(dim_items)
    return scores


ITEMIZE_PROMPT = """
Evaluate this embedding approach by listing specific observations across 4 dimensions:
- intent_capture: Does it capture what the person is trying to accomplish?
- cognitive_state: Does it read focus, confusion, exploration, review?
- specificity: Could this description distinguish this frame from similar ones?
- noise_resistance: Does it ignore window chrome, themes, irrelevant UI?

For each observation, assign an impact score:
  +5 exceptional, +3 high, +2 good, +1 small
  -1 small negative, -2 notable gap, -3 serious problem, -5 critical

Return ONLY valid JSON:
{{"items": [
  {{"dimension": "intent_capture", "observation": "...", "impact": 3}},
  {{"dimension": "noise_resistance", "observation": "...", "impact": -2}}
]}}

Rules:
- 2-5 items per dimension (8-20 items total)
- Cite specific evidence from the approach text
- Don't double-count across dimensions

Original frame: \"\"\"{frame}\"\"\"

Proposed approach:
\"\"\"{approach}\"\"\"
"""


# ─── Base Agent (inline for self-contained skill) ─────────────────

class Agent:
    def __init__(self, name, role, goal):
        self.name   = name
        self.role   = role
        self.goal   = goal
        self.redis  = None  # set in start()
        self.memory = []
        self.current_task  = "idle"
        self.last_score    = 0.0
        self.impact_items  = []
        self.top_positive  = ""
        self.top_negative  = ""

    async def start(self):
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)

    def get_status(self):
        return {
            "agent":        self.name,
            "what":         self.current_task,
            "goal":         self.goal,
            "score":        self.last_score,
            "top_positive": self.top_positive,
            "top_negative": self.top_negative,
            "help_needed":  "struggling" if self.last_score < 0.5 else "ok",
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }

    def build_prompt(self):
        base = f"You are {self.name}, a {self.role}.\nGoal: {self.goal}"
        if not self.memory:
            return base
        recent = self.memory[-3:]
        history = "\n".join(
            f"Score {m['score']:.2f}: best={m['top_positive']} | worst={m['top_negative']}"
            for m in recent
        )
        return f"{base}\n\nRecent performance (use to improve):\n{history}"

    def ask(self, prompt):
        return ollama.chat(prompt=prompt, system=self.build_prompt(), max_tokens=512)

    async def publish(self, stream, to, msg_type, payload):
        msg = {
            "id":        f"msg_{uuid.uuid4().hex[:6]}",
            "from":      self.name,
            "to":        to,
            "type":      msg_type,
            "payload":   payload,
            "context":   self.get_status(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.redis.xadd(stream, {"data": json.dumps(msg)})
        print(f"  [{self.name}] → {stream} ({msg_type})")
        return msg

    async def consume(self, stream, group, handler):
        try:
            await self.redis.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            pass
        print(f"[{self.name}] Listening on {stream}")
        while True:
            results = await self.redis.xreadgroup(
                groupname=group, consumername=self.name,
                streams={stream: ">"}, count=1, block=2000,
            )
            if results:
                for _, messages in results:
                    for msg_id, data in messages:
                        await handler(json.loads(data["data"]))
                        await self.redis.xack(stream, group, msg_id)

    async def heartbeat(self):
        while True:
            await self.redis.xadd("agent:health", {
                "data": json.dumps({"type": "heartbeat", **self.get_status()})
            })
            await asyncio.sleep(20)


# ─── Agent 1: Frame Capture ───────────────────────────────────────

class FrameCaptureAgent(Agent):
    """
    Captures the screen (or generates a demo frame).
    Publishes raw frame data to screen:frames.
    It does NOT embed — that's the embedder's job.
    """
    def __init__(self, demo=False, interval=8):
        super().__init__(
            name = "frame-capture",
            role = "Screen Frame Capture Agent",
            goal = "Capture clear, useful frames for the embedding pipeline",
        )
        self.demo     = demo
        self.interval = interval
        self._idx     = 0

    async def run(self):
        await self.start()
        asyncio.create_task(self.heartbeat())
        print(f"[frame-capture] Starting {'DEMO' if self.demo else 'LIVE'} — interval {self.interval}s")
        while True:
            self.current_task = "capturing frame"
            description, img_b64 = await self._capture()
            await self.publish(
                stream  = "screen:frames",
                to      = "embedder",
                msg_type= "frame",
                payload = {"description": description, "image_b64": img_b64 or "", "has_image": bool(img_b64)},
            )
            await asyncio.sleep(self.interval)

    async def _capture(self):
        if self.demo:
            desc = DEMO_FRAMES[self._idx % len(DEMO_FRAMES)]
            self._idx += 1
            return desc, None
        try:
            import pyautogui
            ss = pyautogui.screenshot()
            buf = BytesIO()
            ss.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            desc = ollama.chat(
                prompt="Describe what the person is doing on this screen in 2-3 sentences. "
                       "Include: tool, task, cognitive state, work stage.",
                image_b64=b64,
            )
            return desc, b64
        except Exception as e:
            print(f"[frame-capture] Falling back to demo: {e}")
            desc = DEMO_FRAMES[self._idx % len(DEMO_FRAMES)]
            self._idx += 1
            return desc, None


# ─── Agent 2: Embedder ────────────────────────────────────────────

class EmbedderAgent(Agent):
    def __init__(self):
        super().__init__(
            name = "embedder",
            role = "Screen Embedding Strategist",
            goal = "Produce embedding descriptions that best capture human work intent, cognitive state, and work stage",
        )

    async def run(self):
        await self.start()
        asyncio.create_task(self.heartbeat())
        await self.consume("screen:frames", "embedders", self._handle_frame)

    async def _handle_frame(self, message):
        description = message["payload"]["description"]
        self.current_task = f"Proposing embedding approach for: {description[:50]}..."

        approach_text = self.ask(
            f"Frame description: \"{description}\"\n\n"
            f"Propose the best way to embed this frame for a model learning human work behavior.\n"
            f"Your approach should capture: intent, cognitive state, work stage, tool context.\n\n"
            f"APPROACH: [describe your embedding strategy]\n"
            f"EMBEDDING_TEXT: [the actual text you would embed — max 150 words]\n"
            f"REASONING: [why this approach will produce better vector representations]"
        )

        await self.publish(
            stream   = "screen:approaches",
            to       = "approach-tester",
            msg_type = "approach_proposal",
            payload  = {
                "source_description": description,
                "approach_text":      approach_text,
            },
        )


# ─── Agent 3: Approach Tester (Two-Pass Scoring) ─────────────────

class ApproachTesterAgent(Agent):
    def __init__(self):
        super().__init__(
            name = "approach-tester",
            role = "Embedding Approach Evaluator",
            goal = "Honestly score embedding approaches using itemized impact observations so the swarm improves",
        )

    async def run(self):
        await self.start()
        asyncio.create_task(self.heartbeat())
        await self.consume("screen:approaches", "scorers", self._handle_approach)

    async def _handle_approach(self, message):
        payload      = message["payload"]
        source       = payload["source_description"]
        approach_txt = payload["approach_text"]
        self.current_task = "Scoring approach (two-pass)"

        # Pass 1: Itemize observations (LLM call)
        # Needs more tokens than the default 512 — 8-20 items × ~30 tokens each
        items_response = ollama.chat(
            prompt=ITEMIZE_PROMPT.format(frame=source, approach=approach_txt),
            system=self.build_prompt(),
            max_tokens=1024,
        )

        try:
            clean = items_response.strip().strip("```json").strip("```").strip()
            parsed = json.loads(clean)
            items = parsed.get("items", [])
        except Exception:
            items = []

        # Pass 2: Derive score (pure math)
        score = derive_score(items)
        dim_scores = dimension_scores(items)

        # Extract top positive/negative for status and feedback loop
        positives = sorted([i for i in items if i["impact"] > 0], key=lambda x: -x["impact"])
        negatives = sorted([i for i in items if i["impact"] < 0], key=lambda x: x["impact"])

        self.impact_items = items
        self.last_score   = score["overall"]
        self.top_positive = positives[0]["observation"] if positives else ""
        self.top_negative = negatives[0]["observation"] if negatives else ""

        # Memory for build_prompt feedback loop
        self.memory.append({
            "score":        score["overall"],
            "top_positive": self.top_positive,
            "top_negative": self.top_negative,
        })

        # Publish to screen:results (refiner reads this)
        await self.publish(
            stream   = "screen:results",
            to       = "broadcast",
            msg_type = "score",
            payload  = {
                "score":            score,
                "dimension_scores": dim_scores,
                "items":            items,
                "source":           source[:80],
            },
        )

        print(f"  [approach-tester] Score: {score['overall']:.2f} "
              f"(net={score['net_impact']}, items={score['total_items']}) "
              f"| {self.top_positive[:50]}")


# ─── Entry point ──────────────────────────────────────────────────

async def run_all(demo):
    print(f"\n🐝  Swarm starting {'[DEMO]' if demo else '[LIVE]'}")
    print(f"   Model: {ollama.OLLAMA_MODEL} (local via Ollama)")
    print("   screen:frames → embedder → approach-tester → scores\n")

    capturer = FrameCaptureAgent(demo=demo, interval=6)
    embedder = EmbedderAgent()
    tester   = ApproachTesterAgent()

    await asyncio.gather(capturer.run(), embedder.run(), tester.run())


async def run_one(agent_name, demo):
    agents = {
        "capture":  FrameCaptureAgent(demo=demo),
        "embedder": EmbedderAgent(),
        "tester":   ApproachTesterAgent(),
    }
    a = agents.get(agent_name)
    if not a:
        print(f"Unknown agent. Choose: {list(agents.keys())}")
        return
    await a.run()


if __name__ == "__main__":
    if not ollama.is_available():
        print("❌  Ollama is not running. Start it with: ollama serve"); exit(1)

    print(f"Model: {ollama.OLLAMA_MODEL} (local via Ollama)")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  default="demo", choices=["demo", "live"])
    parser.add_argument("--agent", default="all")
    args = parser.parse_args()

    try:
        if args.agent == "all":
            asyncio.run(run_all(args.mode == "demo"))
        else:
            asyncio.run(run_one(args.agent, args.mode == "demo"))
    except KeyboardInterrupt:
        print("\n👋  Swarm stopped.")
