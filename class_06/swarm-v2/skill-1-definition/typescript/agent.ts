/**
 * skill-1-definition/typescript/agent.ts
 * ────────────────────────────────────────
 * SKILL 1: Agent Definition (TypeScript)
 *
 * Exact same contract as the Python agent:
 * - get_status() answers the six questions
 * - add_to_memory() stores scored results (two-pass impact items)
 * - build_system_prompt() injects history
 * - act() calls the local model with the built prompt
 *
 * All inference is local via Ollama — no cloud API calls.
 *
 * Usage:
 *   npm install && npx tsx agent.ts
 */

// ── Config ──────────────────────────────────────────────────────

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
const OLLAMA_MODEL    = process.env.OLLAMA_MODEL ?? "qwen3.5:latest";

// ── Types ─────────────────────────────────────────────────────────

interface ImpactItem {
  dimension:   string;
  observation: string;
  impact:      number;   // +5, +3, +2, +1, -1, -2, -3, -5
}

interface DerivedScore {
  overall:            number;
  net_impact:         number;
  total_items:        number;
  normalized_impact:  number;
  raw_score:          number;
}

interface ScoreResult {
  items:    ImpactItem[];
  derived?: DerivedScore;    // optional — will be computed if missing
}

interface AgentStatus {
  agent:         string;
  what:          string;
  goal:          string;
  why:           string;
  how:           string;
  score:         number;
  top_positive:  string;
  top_negative:  string;
  help_needed:   string;
  timestamp:     string;
}

interface MemoryEntry {
  task:          string;
  result:        string;
  score:         number;
  top_positive:  string;
  top_negative:  string;
  impact_items:  ImpactItem[];
}

// ── Two-Pass Scoring ────────────────────────────────────────────
// Pass 1 (LLM): itemize observations with +/- impact scores
// Pass 2 (math): net_impact / √(items) → normalized 0-1 score

function deriveScore(items: ImpactItem[]): DerivedScore {
  if (items.length === 0) {
    return { overall: 0.5, net_impact: 0, total_items: 0,
             normalized_impact: 0, raw_score: 50 };
  }
  const netImpact = items.reduce((sum, i) => sum + i.impact, 0);
  const totalItems = items.length;
  const normalizedImpact = netImpact / Math.sqrt(totalItems);
  const rawScore = Math.max(0, Math.min(100, 50 + normalizedImpact * 8.0));
  const overall = Math.round((rawScore / 100) * 1000) / 1000;
  return { overall, net_impact: netImpact, total_items: totalItems,
           normalized_impact: Math.round(normalizedImpact * 1000) / 1000,
           raw_score: Math.round(rawScore * 10) / 10 };
}

// ── Ollama Client ────────────────────────────────────────────────

async function ollamaChat(
  prompt: string,
  system: string = "",
  model: string = OLLAMA_MODEL,
  maxTokens: number = 1024,
  imageB64?: string,
): Promise<string> {
  const messages: Array<{role: string; content: string; images?: string[]}> = [];

  if (system) {
    messages.push({ role: "system", content: system });
  }

  const userMsg: {role: string; content: string; images?: string[]} = {
    role: "user",
    content: prompt,
  };
  if (imageB64) {
    userMsg.images = [imageB64];
  }
  messages.push(userMsg);

  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: { num_predict: maxTokens },
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  return data.message?.content ?? "";
}

async function ollamaIsAvailable(): Promise<boolean> {
  try {
    const r = await fetch(`${OLLAMA_BASE_URL}/api/tags`, { signal: AbortSignal.timeout(5000) });
    return r.ok;
  } catch {
    return false;
  }
}

// ── Agent Class ───────────────────────────────────────────────────

class Agent {
  private memory: MemoryEntry[] = [];
  private memoryLimit = 5;

  // Live state
  private currentTask  = "idle";
  private reasoning    = "";
  private approach     = "";
  private lastScore    = 0;
  private topPositive  = "";
  private topNegative  = "";

  constructor(
    private name: string,
    private role: string,
    private goal: string,
  ) {}

  // ── Six Questions ───────────────────────────────────────────────

  getStatus(): AgentStatus {
    return {
      agent:         this.name,
      what:          this.currentTask,
      goal:          this.goal,
      why:           this.reasoning,
      how:           this.approach,
      score:         this.lastScore,
      top_positive:  this.topPositive,
      top_negative:  this.topNegative,
      help_needed:   this.assessHelp(),
      timestamp:     new Date().toISOString(),
    };
  }

  private assessHelp(): string {
    if (this.lastScore === 0) return "No score yet — just getting started";
    if (this.lastScore < 0.4) return `Struggling (${this.lastScore.toFixed(2)}) — top issue: ${this.topNegative}`;
    if (this.lastScore < 0.7) return `Progress (${this.lastScore.toFixed(2)}) — working on: ${this.topNegative}`;
    return `Performing well (${this.lastScore.toFixed(2)}) — no help needed`;
  }

  // ── Memory ──────────────────────────────────────────────────────

  addToMemory(task: string, result: string, score: ScoreResult): void {
    const items = score.items;
    const derived = score.derived ?? deriveScore(items);

    const positives = items.filter(i => i.impact > 0).sort((a, b) => b.impact - a.impact);
    const negatives = items.filter(i => i.impact < 0).sort((a, b) => a.impact - b.impact);

    const entry: MemoryEntry = {
      task,
      result:        result.slice(0, 400),
      score:         derived.overall,
      top_positive:  positives[0]?.observation ?? "",
      top_negative:  negatives[0]?.observation ?? "",
      impact_items:  items,
    };

    this.memory.push(entry);
    if (this.memory.length > this.memoryLimit) this.memory.shift();

    // Update live state
    this.lastScore   = entry.score;
    this.topPositive = entry.top_positive;
    this.topNegative = entry.top_negative;
  }

  buildSystemPrompt(): string {
    const base = [
      `You are ${this.name}, a ${this.role}.`,
      `Goal: ${this.goal}`,
      ``,
      `After completing any task, structure your response as:`,
      `WHAT: [what you did]`,
      `HOW: [how you approached it]`,
      `RESULT: [the output]`,
      `CONFIDENCE: [0.0-1.0]`,
      `HELP: [what would make your next attempt better]`,
    ].join("\n");

    if (this.memory.length === 0) return base;

    const history = this.memory.slice(-3).map(m =>
      `Task: ${m.task.slice(0, 80)}\n` +
      `Score: ${m.score.toFixed(2)} | ` +
      `Best: ${m.top_positive} | ` +
      `Worst: ${m.top_negative}`
    ).join("\n\n");

    return `${base}\n\nYour recent performance — use this to improve:\n${history}`;
  }

  // ── LLM ─────────────────────────────────────────────────────────

  async ask(prompt: string, imageB64?: string): Promise<string> {
    return ollamaChat(prompt, this.buildSystemPrompt(), OLLAMA_MODEL, 1024, imageB64);
  }

  async act(task: string, imageB64?: string): Promise<string> {
    this.currentTask = task;
    return this.ask(task, imageB64);
  }
}

// ── Demo ─────────────────────────────────────────────────────────

async function main() {
  if (!(await ollamaIsAvailable())) {
    console.error("Ollama is not running at", OLLAMA_BASE_URL);
    console.error("Start it with: ollama serve");
    process.exit(1);
  }
  console.log(`Model: ${OLLAMA_MODEL} (local via Ollama)`);

  const agent = new Agent(
    "embedder",
    "Screen Embedding Strategist",
    "Produce embedding descriptions that best capture human work intent from screen frames",
  );

  const rounds: Array<{ task: string; score: ScoreResult }> = [
    {
      task: "Describe this screen frame for embedding: Excel spreadsheet open with multiple tabs and a PDF in split screen",
      score: { items: [
        { dimension: "intent_capture",   observation: "Identified tools correctly",       impact: +1 },
        { dimension: "specificity",      observation: "Concise output",                   impact: +1 },
        { dimension: "intent_capture",   observation: "Missed cognitive state entirely",  impact: -3 },
        { dimension: "specificity",      observation: "Just listed what's visible",       impact: -2 },
        { dimension: "noise_resistance", observation: "No mention of work stage",         impact: -2 },
      ]}
    },
    {
      task: "Describe this screen frame for embedding: Figma canvas with a landing page design being iterated on",
      score: { items: [
        { dimension: "intent_capture",  observation: "Captured creative design context",     impact: +3 },
        { dimension: "specificity",     observation: "Included tool and task correctly",     impact: +2 },
        { dimension: "cognitive_state", observation: "Noted iteration pattern",              impact: +1 },
        { dimension: "intent_capture",  observation: "Didn't infer intent behind iteration", impact: -2 },
        { dimension: "cognitive_state", observation: "Should note focused creative state",   impact: -1 },
      ]}
    },
    {
      task: "Describe this screen frame for embedding: Slack open with browser showing research articles in background",
      score: { items: [
        { dimension: "intent_capture",  observation: "Inferred context switch accurately", impact: +3 },
        { dimension: "cognitive_state", observation: "Identified research mode",           impact: +3 },
        { dimension: "specificity",     observation: "Distinguished Slack from browser",   impact: +2 },
        { dimension: "noise_resistance",observation: "Focused on work-relevant signals",   impact: +2 },
        { dimension: "cognitive_state", observation: "Could note distraction pattern",     impact: -1 },
      ]}
    },
  ];

  console.log("\n" + "═".repeat(60));
  console.log("  SKILL 1: Agent Definition Demo (TypeScript)");
  console.log("═".repeat(60) + "\n");

  for (let i = 0; i < rounds.length; i++) {
    const r = rounds[i];
    console.log(`── Round ${i + 1} ──────────────────────────────────`);
    console.log(`Task: ${r.task.slice(0, 80)}...`);

    const result = await agent.act(r.task);
    console.log(`\nResponse (first 300 chars):\n  ${result.slice(0, 300)}...`);

    agent.addToMemory(r.task, result, r.score);

    // Show two-pass scoring
    const derived = deriveScore(r.score.items);
    console.log(`\nTwo-pass scoring:`);
    console.log(`  Pass 1 — ${r.score.items.length} impact items:`);
    for (const item of r.score.items) {
      const sign = item.impact > 0 ? "+" : "";
      console.log(`    [${item.dimension}] ${sign}${item.impact}: ${item.observation}`);
    }
    console.log(`  Pass 2 — deriveScore():`);
    console.log(`    net_impact=${derived.net_impact}  √items=${Math.sqrt(r.score.items.length).toFixed(2)}  overall=${derived.overall}`);

    const status = agent.getStatus();
    console.log(`\nSix Questions after round ${i + 1}:`);
    console.log(`  Score:         ${status.score.toFixed(2)}`);
    console.log(`  Top positive:  ${status.top_positive}`);
    console.log(`  Top negative:  ${status.top_negative}`);
    console.log(`  Help needed:   ${status.help_needed}`);
    console.log();
  }
}

main().catch(console.error);
