/**
 * skill-3-monitoring/server.ts
 * ─────────────────────────────
 * SKILL 3: Agent Monitoring
 *
 * Reads four streams to build a live picture of the swarm:
 *   agent:health     → is each agent alive?
 *   screen:frames    → how fast are frames coming in?
 *   screen:approaches→ how many approaches proposed?
 *   screen:results   → what are scores trending?
 *
 * Exposes:
 *   GET /           → HTML dashboard (auto-refresh)
 *   GET /status     → full JSON status
 *   GET /scores     → score history + trend
 *   GET /agents     → agent liveness
 *   GET /streams    → queue depths (how backed up is each stream?)
 *
 * Usage:
 *   npm install && npx tsx server.ts
 *   open http://localhost:3000
 */

import express from "express";
import cors from "cors";
import { createClient, RedisClientType } from "redis";

const app   = express();
const PORT  = process.env.PORT || 3000;
const REDIS = process.env.REDIS_URL || "redis://localhost:6379";

app.use(cors());

// ── State ─────────────────────────────────────────────────────────

interface AgentInfo {
  last_seen:    string;
  score:        number;
  task:         string;
  what_worked:  string;
  what_didnt:   string;
  help_needed:  string;
  status:       "alive" | "stale";
}

interface ScoreEntry {
  ts:      string;
  overall: number;
  worked:  string;
  didnt:   string;
}

const agentStatus: Record<string, AgentInfo> = {};
const scoreHistory: ScoreEntry[] = [];
const streamDepths: Record<string, number> = {};

// ── Redis watchers ────────────────────────────────────────────────

async function startWatching(redis: RedisClientType) {
  // Create consumer groups
  for (const [stream, group] of [
    ["agent:health",       "monitor"],
    ["screen:results",     "monitor"],
  ] as const) {
    try {
      await redis.xGroupCreate(stream, group, "0", { MKSTREAM: true });
    } catch { /* exists */ }
  }

  // Watch health
  watchStream(redis, "agent:health", "monitor", "mon-health", (msg) => {
    const agent = msg.agent as string;
    if (!agent) return;
    agentStatus[agent] = {
      last_seen:   msg.timestamp as string,
      score:       (msg.score as number) ?? 0,
      task:        (msg.what as string) ?? "",
      what_worked: (msg.what_worked as string) ?? "",
      what_didnt:  (msg.what_didnt as string) ?? "",
      help_needed: (msg.help_needed as string) ?? "",
      status:      "alive",
    };
  });

  // Watch scores
  watchStream(redis, "screen:results", "monitor", "mon-scores", (msg) => {
    const score = (msg.payload as any)?.score ?? {};
    scoreHistory.push({
      ts:      (msg.timestamp as string) ?? new Date().toISOString(),
      overall: score.overall ?? 0,
      worked:  score.what_worked ?? "",
      didnt:   score.what_didnt ?? "",
    });
    if (scoreHistory.length > 100) scoreHistory.shift();
  });

  // Poll stream depths every 5s
  setInterval(async () => {
    for (const stream of ["screen:frames", "screen:approaches", "screen:results", "agent:health"]) {
      try {
        streamDepths[stream] = await redis.xLen(stream);
      } catch { /* ignore */ }
    }
  }, 5000);

  // Mark stale agents
  setInterval(() => {
    const now = Date.now();
    for (const [name, info] of Object.entries(agentStatus)) {
      if (now - new Date(info.last_seen).getTime() > 60_000) {
        agentStatus[name].status = "stale";
      }
    }
  }, 10_000);
}

async function watchStream(
  redis: RedisClientType,
  stream: string,
  group: string,
  consumer: string,
  handler: (payload: Record<string, unknown>) => void,
) {
  while (true) {
    try {
      const results = await redis.xReadGroup(
        group, consumer,
        [{ key: stream, id: ">" }],
        { COUNT: 10, BLOCK: 1000 },
      );
      if (results) {
        for (const { messages } of results) {
          for (const { id, message } of messages) {
            const parsed = JSON.parse(message.data);
            handler(parsed);
            await redis.xAck(stream, group, id);
          }
        }
      }
    } catch { await new Promise(r => setTimeout(r, 500)); }
  }
}

// ── Routes ────────────────────────────────────────────────────────

function getStatus() {
  const recent = scoreHistory.slice(-20);
  const avg = recent.length
    ? recent.reduce((s, e) => s + e.overall, 0) / recent.length
    : 0;
  const trend = recent.length > 2 && recent.at(-1)!.overall > recent[0].overall
    ? "↑ improving" : "→ stable";

  return {
    agents:        agentStatus,
    agent_count:   Object.keys(agentStatus).length,
    score_history: recent,
    stream_depths: streamDepths,
    avg_score:     Math.round(avg * 100) / 100,
    trend,
    updated_at:    new Date().toISOString(),
  };
}

app.get("/status", (_req, res) => res.json(getStatus()));
app.get("/agents", (_req, res) => res.json({ agents: agentStatus }));
app.get("/scores", (_req, res) => res.json({ history: scoreHistory.slice(-50), avg_score: getStatus().avg_score, trend: getStatus().trend }));
app.get("/streams", (_req, res) => res.json({ depths: streamDepths }));

// HTML Dashboard
app.get("/", (_req, res) => {
  res.setHeader("Content-Type", "text/html");
  res.send(`<!DOCTYPE html>
<html>
<head>
  <title>Swarm Monitor</title>
  <meta http-equiv="refresh" content="5">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; padding: 24px; }
    h1 { color: #58a6ff; margin-bottom: 4px; }
    .sub { color: #8b949e; font-size: 13px; margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
    .card h2 { color: #58a6ff; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    td, th { padding: 6px 10px; text-align: left; border-bottom: 1px solid #21262d; }
    th { color: #58a6ff; font-size: 11px; text-transform: uppercase; }
    .bar-wrap { background: #21262d; border-radius: 3px; height: 8px; width: 80px; display: inline-block; vertical-align: middle; }
    .bar-fill { background: #3fb950; height: 8px; border-radius: 3px; }
    .alive { color: #3fb950; } .stale { color: #f85149; }
    .score-big { font-size: 32px; color: #3fb950; font-weight: bold; }
    .trend { color: #58a6ff; margin-left: 8px; }
    .depth { color: #e3b341; }
    .help { color: #8b949e; font-size: 11px; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  </style>
</head>
<body>
  <h1>🐝 Swarm Monitor — Screen Embedding Pipeline</h1>
  <p class="sub">Auto-refreshes every 5s &nbsp;|&nbsp; <a href="/status" style="color:#58a6ff">/status JSON</a></p>
  <div id="app">Loading...</div>
  <script>
    async function update() {
      const s = await fetch('/status').then(r => r.json());
      const agents = Object.entries(s.agents || {});
      let html = '<div class="grid">';

      // Agent table
      html += '<div class="card"><h2>Agents</h2><table><tr><th>Name</th><th>Score</th><th>Status</th><th>Help</th></tr>';
      for (const [name, a] of agents) {
        const pct = Math.round(a.score * 100);
        html += \`<tr>
          <td><b>\${name}</b></td>
          <td><span class="bar-wrap"><span class="bar-fill" style="width:\${pct}%"></span></span> \${a.score.toFixed(2)}</td>
          <td class="\${a.status}">\${a.status}</td>
          <td class="help">\${a.help_needed || '—'}</td>
        </tr>\`;
      }
      if (!agents.length) html += '<tr><td colspan=4 style="color:#8b949e">No agents yet...</td></tr>';
      html += '</table></div>';

      // Score + streams
      html += '<div class="card"><h2>Score Trend</h2>';
      html += \`<p class="score-big">\${s.avg_score}<span class="trend">\${s.trend}</span></p>\`;
      html += '<h2 style="margin-top:16px">Stream Depths</h2><table>';
      for (const [stream, depth] of Object.entries(s.stream_depths || {})) {
        html += \`<tr><td>\${stream}</td><td class="depth">\${depth} msgs</td></tr>\`;
      }
      html += '</table></div></div>';

      // Recent scores
      html += '<div class="card"><h2>Recent Scores</h2><table><tr><th>Score</th><th>What Worked</th><th>What Didn\'t</th></tr>';
      const recent = (s.score_history || []).slice(-8).reverse();
      for (const score of recent) {
        const pct = Math.round(score.overall * 100);
        html += \`<tr>
          <td><span class="bar-wrap"><span class="bar-fill" style="width:\${pct}%"></span></span> \${score.overall.toFixed(2)}</td>
          <td style="color:#3fb950;font-size:12px">\${score.worked?.slice(0,60) || '—'}</td>
          <td style="color:#f85149;font-size:12px">\${score.didnt?.slice(0,60) || '—'}</td>
        </tr>\`;
      }
      html += '</table></div>';

      document.getElementById('app').innerHTML = html;
    }
    update();
  </script>
</body>
</html>`);
});

// ── Start ─────────────────────────────────────────────────────────

async function main() {
  const redis = createClient({ url: REDIS }) as RedisClientType;
  await redis.connect();
  console.log("[monitor] Connected to Redis");
  startWatching(redis);

  app.listen(PORT, () => {
    console.log(`\n🟢  Monitor running → http://localhost:${PORT}`);
    console.log(`    JSON status  → http://localhost:${PORT}/status`);
    console.log(`    Streams      → http://localhost:${PORT}/streams\n`);
  });
}

main().catch(console.error);
