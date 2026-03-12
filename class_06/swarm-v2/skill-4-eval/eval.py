"""
skill-4-eval/eval.py
──────────────────────
SKILL 4: Agent Evaluation

All inference is local via Ollama — no cloud API calls.

Evaluation is not just scoring. It's the signal that makes the swarm smarter.

This module shows three eval patterns:
  1. RUBRIC EVAL     — two-pass impact scoring against explicit criteria
  2. COMPARATIVE EVAL — which of two approaches is better?
  3. REGRESSION CHECK — did the swarm get worse?

Two-pass scoring:
  Pass 1 (LLM): Itemize observations with impact scores (+/- 1, 2, 3, or 5)
  Pass 2 (math): net_impact / sqrt(items) → normalized score

The key insight: a few high-impact observations score higher than many
low-impact ones. Quality of insight over quantity of observations.

Usage:
    python eval.py
    python eval.py --mode compare
    python eval.py --mode regression
"""

import argparse
import json
import math
import os
import sys
from dotenv import load_dotenv

# Add parent dir so we can import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import ollama

load_dotenv()


# ─── Impact Scale ────────────────────────────────────────────────

IMPACT_SCALE = {
    "+5": "Exceptional — rare; exactly the kind of insight that makes embeddings useful",
    "+3": "High impact — clearly strong; demonstrates real understanding",
    "+2": "Good enough — solid observation; noticeably good",
    "+1": "Small impact — nice touch; minor positive signal",
    "-1": "Small negative — minor rough edge; not harmful",
    "-2": "Notable gap — should have been addressed",
    "-3": "High negative — serious problem; undermines embedding quality",
    "-5": "Critical — fundamentally wrong; would produce harmful embeddings",
}

DIMENSIONS = [
    "intent_capture",    # Does it capture what the person is trying to accomplish?
    "cognitive_state",   # Does it read focus, confusion, exploration, review?
    "specificity",       # Could this description distinguish this frame from similar ones?
    "noise_resistance",  # Does it ignore window chrome, themes, irrelevant UI?
]


# ─── Pass 2: Score Derivation (pure math, no LLM) ───────────────

def derive_score(items: list[dict]) -> dict:
    """
    Derive a normalized score from impact items.

    net_impact / sqrt(total_items) rewards quality over quantity.
    5 items averaging +3 = net 15, normalized = 15/√5 ≈ 6.7  → score ~0.90
    20 items averaging +1 = net 20, normalized = 20/√20 ≈ 4.5 → score ~0.86
    The focused evaluation wins.
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

    # Scale to 0–100 centered on 50, then to 0.0–1.0
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


# ─── Pass 1: Itemize (LLM call) ─────────────────────────────────

ITEMIZE_PROMPT = """
You are evaluating an embedding strategy for a screen understanding model.
The model is learning to recognize human work behavior from screen frames.

Evaluate this approach by listing specific observations across 4 dimensions:
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

Approach to evaluate:
\"\"\"{approach}\"\"\"

Original frame:
\"\"\"{frame}\"\"\"
"""


# ─── Pattern 1: Rubric Eval (two-pass) ──────────────────────────

def itemize_approach(approach: str, frame: str) -> list[dict]:
    """Pass 1: Ask LLM to produce itemized impact observations."""
    prompt = ITEMIZE_PROMPT.format(approach=approach, frame=frame)

    raw = ollama.chat(
        prompt=prompt,
        system="You are a precise evaluator. Return only valid JSON. No prose.",
        max_tokens=1024,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw.strip())
    return parsed.get("items", [])


def rubric_eval(approach: str, frame: str) -> dict:
    """Two-pass evaluation: itemize (LLM) then derive score (math)."""
    items = itemize_approach(approach, frame)
    score = derive_score(items)
    dim_scores = dimension_scores(items)

    return {
        "score": score,
        "dimension_scores": dim_scores,
        "items": items,
    }


# ─── Pattern 2: Comparative Eval ─────────────────────────────────

def comparative_eval(approach_a: str, approach_b: str, frame: str) -> dict:
    """Compare two approaches head to head."""
    raw = ollama.chat(
        prompt=f"""
Compare these two embedding approaches for the same screen frame.
Return ONLY valid JSON:
{{
  "winner": "A" or "B",
  "margin": 0.0-1.0,
  "why_winner_won": "one sentence",
  "why_loser_lost": "one sentence",
  "what_a_did_better": "one sentence",
  "what_b_did_better": "one sentence"
}}

Frame: \"\"\"{frame}\"\"\"

Approach A: \"\"\"{approach_a}\"\"\"

Approach B: \"\"\"{approach_b}\"\"\"
""",
        system="You are a precise evaluator. Return only valid JSON. No prose.",
        max_tokens=512,
    )
    raw = raw.strip().strip("```json").strip("```").strip()
    return json.loads(raw)


# ─── Pattern 3: Regression Check ─────────────────────────────────

def regression_check(score_history: list[dict]) -> dict:
    """Given the last N scores, determine if the swarm is regressing."""
    if len(score_history) < 3:
        return {"status": "insufficient_data", "recommendation": "need at least 3 scores"}

    scores    = [s["overall"] for s in score_history]
    avg_early = sum(scores[:len(scores)//2]) / (len(scores)//2)
    avg_late  = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
    trend     = avg_late - avg_early

    if trend < -0.1:
        status = "regressing"
        rec    = "Scores dropped significantly. Consider resetting to last known-good approach."
    elif trend < 0:
        status = "slightly_declining"
        rec    = "Slight decline. Review negative impact items for common patterns."
    elif trend < 0.05:
        status = "plateau"
        rec    = "Scores stable but not improving. Try a fundamentally different approach."
    else:
        status = "improving"
        rec    = "Scores improving. Keep current strategy."

    return {
        "status":         status,
        "avg_early":      round(avg_early, 3),
        "avg_late":       round(avg_late, 3),
        "trend":          round(trend, 3),
        "recommendation": rec,
    }


# ─── Demo ─────────────────────────────────────────────────────────

DEMO_FRAME = "Excel spreadsheet open with quarterly revenue data across multiple tabs. PDF report visible in split screen. Person cross-referencing numbers."

APPROACH_A = (
    "Excel environment. Spreadsheet with data. PDF visible. "
    "Multiple tabs open. Person working with numbers."
)

APPROACH_B = (
    "Financial analyst is building a quarterly revenue model — cross-referencing "
    "a PDF report signals a verification/validation cognitive state. Work stage: active analysis. "
    "Focused and methodical. Intent: reconcile data across sources, not create new content."
)

# Simulated impact items for regression demo (no Ollama needed)
SIMULATED_SCORES = [
    {"overall": 0.45, "items": [
        {"dimension": "intent_capture", "observation": "Identified the tool but not the task", "impact": 1},
        {"dimension": "cognitive_state", "observation": "No cognitive signal at all", "impact": -3},
        {"dimension": "specificity", "observation": "Too generic — 'working with numbers'", "impact": -2},
        {"dimension": "noise_resistance", "observation": "Listed tab count — irrelevant detail", "impact": -1},
    ]},
    {"overall": 0.52, "items": [
        {"dimension": "intent_capture", "observation": "Named the activity as data work", "impact": 2},
        {"dimension": "cognitive_state", "observation": "No cognitive state mentioned", "impact": -2},
        {"dimension": "specificity", "observation": "Could apply to any spreadsheet task", "impact": -2},
        {"dimension": "noise_resistance", "observation": "Mentioned 'PDF visible' without purpose", "impact": -1},
    ]},
    {"overall": 0.48, "items": [
        {"dimension": "intent_capture", "observation": "Said 'person working' — too vague", "impact": -2},
        {"dimension": "cognitive_state", "observation": "No cognitive signal", "impact": -2},
        {"dimension": "specificity", "observation": "Short and concise at least", "impact": 1},
        {"dimension": "noise_resistance", "observation": "Didn't describe window chrome", "impact": 1},
    ]},
    {"overall": 0.61, "items": [
        {"dimension": "intent_capture", "observation": "Started mentioning analytical context", "impact": 2},
        {"dimension": "cognitive_state", "observation": "Still no cognitive state", "impact": -2},
        {"dimension": "specificity", "observation": "Referenced revenue specifically", "impact": 2},
        {"dimension": "noise_resistance", "observation": "Listed tools but not their purpose", "impact": -1},
    ]},
    {"overall": 0.55, "items": [
        {"dimension": "intent_capture", "observation": "Mentioned cross-referencing as task", "impact": 2},
        {"dimension": "cognitive_state", "observation": "No focus/stuck/exploring signal", "impact": -2},
        {"dimension": "specificity", "observation": "Work stage mentioned but vague", "impact": 1},
        {"dimension": "noise_resistance", "observation": "Too descriptive not semantic", "impact": -1},
    ]},
]


def main():
    if not ollama.is_available():
        print("❌  Ollama is not running. Start it with: ollama serve"); return

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rubric", choices=["rubric", "compare", "regression"])
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SKILL 4: Evaluation — mode: {args.mode}")
    print(f"  Model: {ollama.OLLAMA_MODEL} (local via Ollama)")
    print(f"{'='*60}\n")

    if args.mode == "rubric":
        print("Two-pass scoring Approach A...\n")
        result = rubric_eval(APPROACH_A, DEMO_FRAME)
        print("Impact items:")
        for item in result["items"]:
            sign = "+" if item["impact"] > 0 else ""
            print(f"  [{item['dimension']}] ({sign}{item['impact']}) {item['observation']}")
        print(f"\nDerived score: {json.dumps(result['score'], indent=2)}")
        print(f"\nPer-dimension:")
        for dim, ds in result["dimension_scores"].items():
            print(f"  {dim}: {ds['overall']:.2f} (net={ds['net_impact']}, items={ds['total_items']})")

    elif args.mode == "compare":
        print("Comparing Approach A vs Approach B...\n")
        print(f"A: {APPROACH_A[:100]}...")
        print(f"B: {APPROACH_B[:100]}...\n")
        result = comparative_eval(APPROACH_A, APPROACH_B, DEMO_FRAME)
        print(json.dumps(result, indent=2))

    elif args.mode == "regression":
        print("Running regression check on simulated score history...\n")
        result = regression_check(SIMULATED_SCORES)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
