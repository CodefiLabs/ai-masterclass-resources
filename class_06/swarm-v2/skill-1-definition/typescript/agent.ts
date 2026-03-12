/**
 * skill-1-definition/typescript/agent.ts
 * ────────────────────────────────────────
 * SKILL 1: Agent Definition (TypeScript)
 *
 * Exact same contract as the Python agent:
 * - get_status() answers the six questions
 * - add_to_memory() stores scored results
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

interface AgentStatus {
  agent:        string;
  what:         string;   // What are you doing?
  goal:         string;   // What's your goal?
  why:          string;   // Why?
  how:          string;   // How?
  score:        number;   // Last score 0-1
  what_worked:  string;   // what's working + strongest parts + what to keep
  what_didnt:   string;   // what's not working + weakest parts + what to change
  what_missing: string;   // what should have been included but wasn't
  help_needed:  string;   // How can I help?
  timestamp:    string;
}

interface ScoreResult {
  overall:      number;
  what_worked:  string;
  what_didnt:   string;
  what_missing: string;
}

interface MemoryEntry {
  task:         string;
  result:       string;
  score:        number;
  what_worked:  string;
  what_didnt:   string;
  what_missing: string;
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
  private whatWorked   = "";
  private whatDidnt    = "";
  private whatMissing  = "";

  constructor(
    private name: string,
    private role: string,
    private goal: string,
  ) {}

  // ── Six Questions ───────────────────────────────────────────────

  getStatus(): AgentStatus {
    return {
      agent:        this.name,
      what:         this.currentTask,
      goal:         this.goal,
      why:          this.reasoning,
      how:          this.approach,
      score:        this.lastScore,
      what_worked:  this.whatWorked,
      what_didnt:   this.whatDidnt,
      what_missing: this.whatMissing,
      help_needed:  this.assessHelp(),
      timestamp:    new Date().toISOString(),
    };
  }

  private assessHelp(): string {
    if (this.lastScore === 0) return "No score yet — just getting started";
    if (this.lastScore < 0.4) return `Struggling (${this.lastScore.toFixed(2)}) — need different approach for: ${this.whatDidnt}`;
    if (this.lastScore < 0.7) return `Progress (${this.lastScore.toFixed(2)}) — could improve: ${this.whatMissing}`;
    return `Performing well (${this.lastScore.toFixed(2)}) — no help needed`;
  }

  // ── Memory ──────────────────────────────────────────────────────

  addToMemory(task: string, result: string, score: ScoreResult): void {
    const entry: MemoryEntry = {
      task,
      result:       result.slice(0, 400),
      score:        score.overall,
      what_worked:  score.what_worked,
      what_didnt:   score.what_didnt,
      what_missing: score.what_missing,
    };

    this.memory.push(entry);
    if (this.memory.length > this.memoryLimit) this.memory.shift();

    // Update live state
    this.lastScore   = entry.score;
    this.whatWorked  = entry.what_worked;
    this.whatDidnt   = entry.what_didnt;
    this.whatMissing = entry.what_missing;
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
      `Worked: ${m.what_worked} | ` +
      `Didn't: ${m.what_didnt} | ` +
      `Missing: ${m.what_missing}`
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
  // Check Ollama is running
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

  const rounds = [
    {
      task: "Describe this screen frame for embedding: Excel spreadsheet open with multiple tabs and a PDF in split screen",
      score: { overall: 0.45, what_worked: "Identified the tools; concise", what_didnt: "Too surface-level — missed cognitive state", what_missing: "No work stage" }
    },
    {
      task: "Describe this screen frame for embedding: Figma canvas with a landing page design being iterated on",
      score: { overall: 0.68, what_worked: "Captured creative context; tool + task included", what_didnt: "Didn't infer enough about person's intent", what_missing: "Should note focused refinement state" }
    },
    {
      task: "Describe this screen frame for embedding: Slack open with browser showing research articles in background",
      score: { overall: 0.82, what_worked: "Inferred context switch; cognitive state captured well", what_didnt: "Could note distraction pattern more — missed distraction signal", what_missing: "" }
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

    const status = agent.getStatus();
    console.log(`\nSix Questions after round ${i + 1}:`);
    console.log(`  Score:       ${status.score.toFixed(2)}`);
    console.log(`  What worked: ${status.what_worked}`);
    console.log(`  What didn't: ${status.what_didnt}`);
    console.log(`  Help needed: ${status.help_needed}`);
    console.log();
  }
}

main().catch(console.error);
