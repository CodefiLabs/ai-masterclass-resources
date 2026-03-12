# Screen Pipeline: Capture → Describe → Store → Compare

This is the domain the swarm exists to serve.
The pipeline runs independently of which agents are active.

All inference is local via Ollama. Memory is managed by Hindsight.

## The Flow

```
pyautogui.screenshot()
        ↓
Ollama (qwen3.5:latest)
  "What is this person doing?"
        ↓
Embedding Text (approach proposed by embedder agent)
        ↓
Hindsight (vectorize-io/hindsight)
  retain(content=description, context=approach)
  Auto-embeds + stores — no manual vector management
        ↓
Similarity Search
  recall(query="Find frames similar to this one")
  "Find all 'blocked/stuck' frames"
  "Find all 'deep analysis' frames"
```

## Why Embeddings?

A raw screenshot is ~500KB. An embedding captures the *meaning* of the screen
in a compact vector.

Two frames of the same person doing focused analytical work in different applications
will have very similar embeddings — even if the screens look completely different.

That's what we're training for.

## Running the Pipeline

```bash
pip install -r requirements.txt
python pipeline.py --mode demo      # no real screen needed
python pipeline.py --mode live      # real capture
python pipeline.py --search "debugging"  # similarity search demo
```

## Memory Store

Hindsight (vectorize-io/hindsight) — runs as a Docker container.
Start with: `docker compose up -d`
Admin UI: `http://localhost:9999`
