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
import hashlib
import json
import math
import os
import sys
import time
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

# ─── Default capture config ──────────────────────────────────────
DEFAULT_CAPTURE_CONFIG = {
    "capture_interval": 2,
    "batch_window": 30,
    "max_frames_per_batch": 15,
}

# ─── Synthetic frames for demo mode ──────────────────────────────
# Organized as realistic 30-second session segments.
# Consecutive frames tell a story — the same activity evolving over time.
DEMO_FRAMES = [
    # Segment 1: Excel data analysis session
    "Excel open. Revenue spreadsheet Q4. Scrolling through rows of monthly data.",
    "Excel. Same spreadsheet. Selecting column C (revenue) to create a chart.",
    "Excel. Chart wizard open. Choosing bar chart type for revenue comparison.",
    "Excel. Bar chart inserted. Adjusting axis labels and title.",
    # Segment 2: Switching to browser for research
    "Browser opened. Navigating to Google Scholar. Search bar focused.",
    "Google Scholar. Typed 'revenue forecasting ML models'. Results loading.",
    "Google Scholar. Reading abstract of first paper. Cursor hovering PDF link.",
    "Google Scholar. Opened PDF in new tab. Skimming introduction section.",
    # Segment 3: Slack interruption and context switch
    "Slack notification appeared. Switching to Slack. Team channel #analytics.",
    "Slack. Reading message from manager asking for Q4 report status update.",
    "Slack. Typing reply: 'Almost done with the charts, will share by EOD.'",
    "Slack. Reply sent. Switching back to Excel tab.",
    # Segment 4: Back to Excel, building the report
    "Excel. Back on revenue spreadsheet. Copying chart to clipboard.",
    "Google Docs opened. Q4 Report template. Pasting chart into results section.",
    "Google Docs. Writing analysis paragraph below the chart. Focused typing.",
    "Google Docs. Formatting text. Adjusting heading styles. Report taking shape.",
    # Segment 5: Notion planning after report
    "Notion opened. Project board for Analytics team. Kanban view.",
    "Notion. Dragging 'Q4 Report' card from 'In Progress' to 'Review'.",
    "Notion. Creating new card: 'Q1 Forecast Model'. Adding description.",
    "Notion. Assigning card to self. Setting due date. Adding ML tag.",
    # Segment 6: Reviewing and wrapping up
    "Google Docs. Re-reading Q4 report. Scrolling to check formatting.",
    "Google Docs. Adding executive summary at top. High-level bullet points.",
    "Gmail opened. Composing email to manager. Attaching Q4 report link.",
    "Gmail. Email sent. Switching to calendar. Checking tomorrow's meetings.",
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
Evaluate this embedding approach for a batch of screen frames (a 30-second session segment) by listing specific observations across 4 dimensions:
- intent_capture: Does it capture what the person is trying to accomplish across the batch?
- cognitive_state: Does it read focus, confusion, exploration, review, or transitions between states?
- specificity: Could this description distinguish this session segment from similar ones?
- noise_resistance: Does it ignore window chrome, themes, irrelevant UI across all frames?

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
- Evaluate how well the approach captures the SEQUENCE of activity, not just individual frames

Original batch of frames: \"\"\"{frame}\"\"\"

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

def _image_hash(img_bytes: bytes) -> str:
    """Fast perceptual-ish hash: downscale to 8x8 grayscale, hash the pixels."""
    try:
        from PIL import Image
        img = Image.open(BytesIO(img_bytes)).convert("L").resize((8, 8), Image.LANCZOS)
        return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        # Fallback: hash the raw bytes
        return hashlib.md5(img_bytes).hexdigest()


class FrameCaptureAgent(Agent):
    """
    Captures the screen every N seconds (default 2s), deduplicates,
    and buffers unique frames over a batch window (default 30s).
    Publishes the entire batch as a single message to screen:frames.
    Reads capture config from Redis key swarm:capture_config.
    """
    def __init__(self, demo=False):
        super().__init__(
            name = "frame-capture",
            role = "Screen Frame Capture Agent",
            goal = "Capture clear, useful frames for the embedding pipeline",
        )
        self.demo = demo
        self._idx = 0
        # Capture config — will be overridden from Redis if available
        self.capture_interval = DEFAULT_CAPTURE_CONFIG["capture_interval"]
        self.batch_window = DEFAULT_CAPTURE_CONFIG["batch_window"]
        self.max_frames_per_batch = DEFAULT_CAPTURE_CONFIG["max_frames_per_batch"]
        # Dedup state
        self._prev_hash = None   # previous frame hash (live) or text (demo)

    async def _load_config(self):
        """Read capture config from Redis, falling back to defaults."""
        try:
            raw = await self.redis.get("swarm:capture_config")
            if raw:
                cfg = json.loads(raw)
                # Clamp values to valid ranges to prevent tight loops or crashes
                self.capture_interval = max(1, min(5, int(cfg.get("capture_interval", self.capture_interval))))
                self.batch_window = max(15, min(60, int(cfg.get("batch_window", self.batch_window))))
                self.max_frames_per_batch = max(5, min(30, int(cfg.get("max_frames_per_batch", self.max_frames_per_batch))))
                print(f"  [frame-capture] Config loaded: interval={self.capture_interval}s, "
                      f"window={self.batch_window}s, max_frames={self.max_frames_per_batch}")
        except Exception as e:
            print(f"  [frame-capture] Config load failed, using defaults: {e}")

    async def run(self):
        await self.start()
        asyncio.create_task(self.heartbeat())
        await self._load_config()
        # Write default config if none exists
        exists = await self.redis.exists("swarm:capture_config")
        if not exists:
            await self.redis.set("swarm:capture_config", json.dumps(DEFAULT_CAPTURE_CONFIG))
        print(f"[frame-capture] Starting {'DEMO' if self.demo else 'LIVE'} — "
              f"interval {self.capture_interval}s, batch window {self.batch_window}s")
        while True:
            await self._run_batch_cycle()

    async def _run_batch_cycle(self):
        """Collect unique frames over one batch window, then publish the batch."""
        # Re-read config at the start of each batch cycle
        await self._load_config()

        batch = []
        window_start = time.monotonic()

        while (time.monotonic() - window_start) < self.batch_window:
            if len(batch) >= self.max_frames_per_batch:
                # Already at max — just wait out the remaining window
                remaining = self.batch_window - (time.monotonic() - window_start)
                if remaining > 0:
                    await asyncio.sleep(remaining)
                break

            self.current_task = f"capturing frame (batch {len(batch)}/{self.max_frames_per_batch})"
            description, img_b64, is_dup = await self._capture_dedup()

            if not is_dup:
                batch.append({
                    "description": description,
                    "image_b64": img_b64 or "",
                    "has_image": bool(img_b64),
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                })
                print(f"  [frame-capture] Buffered frame {len(batch)}: {description[:60]}...")
            else:
                print(f"  [frame-capture] Skipped duplicate frame")

            await asyncio.sleep(self.capture_interval)

        if not batch:
            print(f"  [frame-capture] Empty batch (all duplicates), skipping publish")
            return

        self.current_task = f"publishing batch of {len(batch)} frames"
        await self.publish(
            stream   = "screen:frames",
            to       = "embedder",
            msg_type = "frame_batch",
            payload  = {
                "batch": True,
                "frame_count": len(batch),
                "frames": batch,
                "window_seconds": self.batch_window,
                "capture_interval": self.capture_interval,
            },
        )
        print(f"  [frame-capture] Published batch: {len(batch)} unique frames over {self.batch_window}s window")

    async def _capture_dedup(self):
        """Capture a frame and check if it's a duplicate. Returns (desc, img_b64, is_dup)."""
        if self.demo:
            desc = DEMO_FRAMES[self._idx % len(DEMO_FRAMES)]
            self._idx += 1
            # Demo dedup: simple string comparison
            if desc == self._prev_hash:
                return desc, None, True
            self._prev_hash = desc
            return desc, None, False

        try:
            import pyautogui
            ss = pyautogui.screenshot()
            buf = BytesIO()
            ss.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            # Live dedup: compare image hashes
            current_hash = _image_hash(img_bytes)
            if current_hash == self._prev_hash:
                return "", None, True
            self._prev_hash = current_hash

            b64 = base64.b64encode(img_bytes).decode()
            desc = ollama.chat(
                prompt="Describe what the person is doing on this screen in 2-3 sentences. "
                       "Include: tool, task, cognitive state, work stage.",
                image_b64=b64,
            )
            return desc, b64, False
        except Exception as e:
            print(f"[frame-capture] Falling back to demo: {e}")
            desc = DEMO_FRAMES[self._idx % len(DEMO_FRAMES)]
            self._idx += 1
            if desc == self._prev_hash:
                return desc, None, True
            self._prev_hash = desc
            return desc, None, False


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

    async def _get_current_approach(self) -> str | None:
        """Read the refined approach from Redis (written by the refiner agent)."""
        try:
            return await self.redis.get("swarm:current_approach")
        except Exception:
            return None

    def build_prompt(self):
        """Override to include the current refined approach if available."""
        base = super().build_prompt()
        # The current approach is injected via _handle_frame, not here,
        # because we need async Redis access.
        return base

    async def _handle_frame(self, message):
        payload = message["payload"]

        # Read the current refined approach from the refiner
        current_approach = await self._get_current_approach()
        approach_context = ""
        if current_approach:
            approach_context = (
                f"\n\nCURRENT REFINED APPROACH (from the swarm refiner — follow these instructions):\n"
                f"{current_approach}\n\n"
                f"Use the above approach as your guide. Adapt it to this specific batch.\n"
            )

        # Handle batched frames
        if payload.get("batch"):
            frames = payload.get("frames", [])
            frame_count = payload.get("frame_count", len(frames))
            window_seconds = payload.get("window_seconds", 30)
            self.current_task = f"Analyzing batch of {frame_count} frames ({window_seconds}s window)"

            # Build a numbered list of frame descriptions for the LLM
            frame_list = "\n".join(
                f"  [{i+1}/{frame_count}] ({f.get('captured_at', '?')}): {f.get('description', '(no description)')}"
                for i, f in enumerate(frames)
            )

            # Build the batch source description for scoring downstream
            source_description = f"Batch of {frame_count} frames over {window_seconds}s:\n{frame_list}"

            batch_prompt = (
                f"You are analyzing a sequence of {frame_count} screen captures taken over {window_seconds} seconds.\n"
                f"This represents a continuous session segment of someone working.\n\n"
                f"SCREEN FRAMES (in chronological order):\n{frame_list}\n\n"
                f"{approach_context}"
                f"Analyze the ENTIRE sequence to understand:\n"
                f"1. OVERALL ACTIVITY: What was the person doing across these {window_seconds} seconds?\n"
                f"2. TRANSITIONS: What context switches or tool changes happened?\n"
                f"3. COGNITIVE FLOW: Were they focused, exploring, reviewing, or switching gears?\n"
                f"4. INTENT ARC: What's the overarching goal threading these frames together?\n\n"
                f"Propose the best way to embed this session segment for a model learning human work behavior.\n"
                f"Your approach should capture the narrative arc, not just individual frames.\n\n"
                f"APPROACH: [describe your embedding strategy for this session segment]\n"
                f"EMBEDDING_TEXT: [the actual text you would embed — max 250 words, capturing the full arc]\n"
                f"REASONING: [why this approach captures the session better than frame-by-frame embedding]"
            )

            # Use ollama.chat() directly with more tokens for batch analysis
            approach_text = ollama.chat(
                prompt=batch_prompt,
                system=super().build_prompt(),
                max_tokens=1024,
            )

            await self.publish(
                stream   = "screen:approaches",
                to       = "approach-tester",
                msg_type = "approach_proposal",
                payload  = {
                    "source_description": source_description,
                    "approach_text":      approach_text,
                    "batch": True,
                    "frame_count": frame_count,
                },
            )
        else:
            # Legacy single-frame handling (backward compatibility)
            description = payload.get("description", "")
            self.current_task = f"Proposing embedding approach for: {description[:50]}..."

            single_prompt = (
                f"Frame description: \"{description}\"\n\n"
                f"{approach_context}"
                f"Propose the best way to embed this frame for a model learning human work behavior.\n"
                f"Your approach should capture: intent, cognitive state, work stage, tool context.\n\n"
                f"APPROACH: [describe your embedding strategy]\n"
                f"EMBEDDING_TEXT: [the actual text you would embed — max 150 words]\n"
                f"REASONING: [why this approach will produce better vector representations]"
            )

            approach_text = self.ask(single_prompt)

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
        is_batch     = payload.get("batch", False)
        frame_count  = payload.get("frame_count", 1)
        self.current_task = f"Scoring approach (two-pass, {'batch of ' + str(frame_count) if is_batch else 'single frame'})"

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
    print("   screen:frames → embedder → approach-tester → scores")
    print(f"   Capture: every {DEFAULT_CAPTURE_CONFIG['capture_interval']}s, "
          f"batch window {DEFAULT_CAPTURE_CONFIG['batch_window']}s, "
          f"max {DEFAULT_CAPTURE_CONFIG['max_frames_per_batch']} frames/batch\n")

    capturer = FrameCaptureAgent(demo=demo)
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
