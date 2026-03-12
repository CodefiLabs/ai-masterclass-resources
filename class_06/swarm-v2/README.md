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
| Agents | Ollama (qwen3.5:latest) | Local LLM does the reasoning |
| Message bus | Redis Streams | Fast, persistent, consumer groups |
| Memory | Hindsight (vectorize-io) | Auto-embeds, stores, and retrieves memories |
| Monitoring | Redis XLEN + Express API | See queue depth and scores |
| PHP layer | Laravel Queue + Predis | Real parity, not stubs |

---

## Quick Start

```bash
# 0. Prerequisites — Ollama must be running with the model pulled
ollama serve
ollama pull qwen3.5:latest

# 1. Start infrastructure (Redis + Hindsight)
docker compose up -d

# 2. Python agents (the core)
cd skill-1-definition/python
cp .env.example .env
pip install -r requirements.txt
python agent.py

# 3. Run the full swarm
cd ../../skill-2-orchestration
python run_swarm.py --mode demo

# 4. Watch it
open http://localhost:3000         # monitoring dashboard
open http://localhost:9999         # Hindsight admin UI
```
