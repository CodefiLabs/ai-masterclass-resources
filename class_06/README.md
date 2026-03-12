# Class 06 — Agent Swarms

The swarm-v2 teaching codebase from the final class of the AI Masterclass series. A multi-agent swarm that runs entirely locally — no cloud API calls, no data leaving the machine.

## Stack

- **Inference:** Ollama (`qwen3.5:latest`)
- **Memory:** Hindsight (vectorize-io)
- **Coordination:** Redis Streams
- **Languages:** Python, TypeScript, PHP (skill-1 has all three as a teaching artifact)

## Structure

```
swarm-v2/
├── shared/              # Ollama client, Hindsight client, stream config
├── skill-1-definition/  # The agent contract — same agent in Python, TypeScript, PHP
├── skill-2-orchestration/  # run_swarm.py — boots agents, manages pipeline
├── skill-3-monitoring/  # TypeScript health dashboard (Redis Stream consumer)
├── skill-4-eval/        # Two-pass impact scoring (itemize → derive)
├── skill-5-refining/    # Evidence-weighted feedback loop
├── screen-pipeline/     # Screen capture → embedding pipeline
├── docker-compose.yml   # Redis + Hindsight
└── README.md
```

## Running the Swarm

```bash
# Prerequisites
ollama serve                          # Local LLM inference
docker compose up -d                  # Redis + Hindsight

# Demo mode (simulated captures, no screen access needed)
cd swarm-v2/skill-2-orchestration
cp .env.example .env
pip install -r requirements.txt
python run_swarm.py --mode demo
```

## Key Concepts

- **Pipeline pattern:** One-way flow via Redis Streams — agents don't know upstream/downstream
- **Two-pass scoring:** Pass 1 (LLM) itemizes observations with +/- impact scores. Pass 2 (math) derives normalized score via `net_impact / √(items)`
- **The help loop:** OBSERVE → PROPOSE → SCORE → REINFORCE
- **12-Factor Agent Principles:** Language-agnostic contract that any agent must satisfy
