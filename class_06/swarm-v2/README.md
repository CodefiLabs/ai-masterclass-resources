# AI Swarm Starter Kit — Class 6
### AI for Developers Masterclass

---

## The Problem This Swarm Solves

You have a model that watches someone's screen — anyone's screen — and tries to understand what they're doing. A designer in Figma. An analyst in Excel. A support rep in Zendesk. A developer in VS Code.
Each frame gets converted to an embedding — a vector that captures the *meaning* of the screen moment.
Over time, the model should get better at recognizing patterns: deep focus, context switching, getting stuck, multitasking across tools.

**The swarm's job is to make that model better.** Agents capture frames, propose better embedding strategies, test them, score the results, and feed the winners back into the next round.

This is reinforced learning in practice — not on toy data, but on real human work behavior.

**Everything runs locally.** No cloud API calls, no data leaves your machine.

---

## The Agent Help Loop

Every agent in this swarm runs the same cycle:

```
1. OBSERVE    — What's happening on screen right now?
2. QUESTION   — What are you doing / goal / why / how?
3. PROPOSE    — What approach might improve the embedding?
4. TEST       — Research → Plan → Implement → Verify
5. SCORE      — What worked / didn't / missing / strong / weak?
6. REINFORCE  — Feed winners back. Discard losers.
```

---

## Five Skills — Five Folders

```
skill-1-definition/     ← What is an agent? Define one in Python, TS, PHP
skill-2-orchestration/  ← How do agents coordinate via message bus?
skill-3-monitoring/     ← How do you see what the swarm is doing?
skill-4-eval/           ← How do you score outputs honestly?
skill-5-refining/       ← How does the swarm get better over time?
screen-pipeline/        ← The domain: capture → describe → store → compare
shared/                 ← Ollama + Hindsight clients, constants
```

Each skill folder is self-contained. Learn them in order or jump to what you need.

---

## The Six Agent Questions

Every agent must be able to answer these at any moment:

| Question | Why It Matters |
|----------|---------------|
| What are you doing? | Prevents silent stalls |
| What's your goal? | Keeps agent on track |
| Why are you doing it? | Enables peers to challenge the approach |
| How are you approaching it? | Surfaces method for scoring |
| What worked / didn't / missing / strong / weak? | The reinforcement signal |
| How can I help? | Opens the collaboration channel |

---

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| Agents | Ollama (qwen3.5:2b) | Local LLM does the reasoning — 2b is fast enough; larger models hit timeout |
| Message bus | Redis Streams | Fast, persistent, consumer groups |
| Memory | Hindsight (vectorize-io) | Auto-embeds, stores, and retrieves memories |
| Monitoring | Redis XLEN + Express API | See queue depth and scores |
| PHP layer | Laravel Queue + Predis | Real parity, not stubs |

> **Model note:** `qwen3.5:latest` (the 7b) is a thinking model — its reasoning tokens count against `num_predict`.
> At ~1s/token it reliably exceeds the 120s timeout in `shared/ollama.py`. Use `qwen3.5:2b` instead.
> If you want to run the larger model, bump `timeout=120` to `timeout=600` in `shared/ollama.py`.

---

## Quick Start

```bash
# 0. Prerequisites — Ollama must be running with the model pulled
ollama serve
ollama pull qwen3.5:2b

# 1. Start infrastructure (Redis + Hindsight)
docker compose up -d

# 2. Create a shared Python virtualenv (system pip is externally managed on macOS)
python3 -m venv .venv
source .venv/bin/activate
pip install -r skill-1-definition/python/requirements.txt \
            -r skill-2-orchestration/requirements.txt \
            -r skill-4-eval/requirements.txt \
            -r skill-5-refining/requirements.txt \
            -r screen-pipeline/requirements.txt

# 3. Copy env files (defaults point to localhost Ollama + Redis)
for dir in skill-1-definition/python skill-2-orchestration skill-4-eval skill-5-refining screen-pipeline; do
  cp $dir/.env.example $dir/.env
done

# 4. Run the full swarm (from repo root, venv active)
PYTHONUNBUFFERED=1 .venv/bin/python3 skill-2-orchestration/run_swarm.py --mode demo

# 5. Watch it
#   monitoring dashboard runs on port 3001 to avoid conflicts (port 3000 is commonly taken)
cd skill-3-monitoring && npm install && PORT=3001 npx tsx server.ts &
open http://localhost:3001         # monitoring dashboard
open http://localhost:9999         # Hindsight admin UI
open http://localhost:8001         # RedisInsight UI
```

> **Note:** The monitoring server listens on port 3000 by default but port 3000 is frequently
> used by other local dev servers. Use `PORT=3001` (or any free port) to avoid conflicts.
