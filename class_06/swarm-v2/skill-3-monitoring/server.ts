/**
 * skill-3-monitoring/server.ts
 * ─────────────────────────────
 * SKILL 3: Agent Monitoring
 *
 * Reads all four streams and builds a live picture of the swarm:
 *   screen:frames      → frame descriptions from capture agent
 *   screen:approaches  → embedding proposals from embedder
 *   screen:results     → scored results with dimension breakdowns
 *   agent:health       → heartbeats from all agents
 *
 * Endpoints:
 *   GET /          → HTML dashboard (polls /feed every 2s)
 *   GET /feed      → unified activity log (last 60 events)
 *   GET /status    → aggregated status JSON
 *   GET /streams   → stream depths
 *
 * Usage:
 *   npm install && PORT=3002 npx tsx server.ts
 */

import express from "express";
import cors from "cors";
import { createClient, RedisClientType } from "redis";

const app   = express();
const PORT  = process.env.PORT || 3002;
const REDIS = process.env.REDIS_URL || "redis://localhost:6379";

app.use(cors());

// ── Types ──────────────────────────────────────────────────────────

type EventKind = "frame" | "approach" | "score" | "heartbeat";

interface FeedEvent {
  id:        string;
  ts:        string;
  kind:      EventKind;
  agent:     string;
  // frame
  frame_description?: string;
  // approach
  source_description?: string;
  approach_text?: string;
  // score
  overall?: number;
  raw_score?: number;
  net_impact?: number;
  total_items?: number;
  items?: Array<{ dimension: string; observation: string; impact: number }>;
  dimension_scores?: Record<string, { overall: number; net_impact: number }>;
  // heartbeat
  task?: string;
  score?: number;
  help_needed?: string;
  top_positive?: string;
  top_negative?: string;
}

interface AgentInfo {
  last_seen:    string;
  score:        number;
  task:         string;
  top_positive: string;
  top_negative: string;
  help_needed:  string;
  status:       "alive" | "stale";
}

// ── State ──────────────────────────────────────────────────────────

const feed:         FeedEvent[]                   = [];
const agentStatus:  Record<string, AgentInfo>     = {};
const streamDepths: Record<string, number>        = {};
const scoreHistory: Array<{ ts: string; overall: number }> = [];

function pushEvent(ev: FeedEvent) {
  feed.unshift(ev); // newest first
  if (feed.length > 120) feed.pop();
}

// ── Redis watchers ─────────────────────────────────────────────────

async function startWatching(redis: RedisClientType) {
  const groups: [string, string][] = [
    ["screen:frames",     "monitor"],
    ["screen:approaches", "monitor"],
    ["screen:results",    "monitor"],
    ["agent:health",      "monitor"],
  ];

  for (const [stream, group] of groups) {
    try {
      await redis.xGroupCreate(stream, group, "0", { MKSTREAM: true });
    } catch { /* exists */ }
  }

  // screen:frames
  watchStream(redis, "screen:frames", "monitor", "mon-frames", (msg) => {
    const p = msg.payload as any;
    pushEvent({
      id:                msg.id as string ?? crypto.randomUUID(),
      ts:                msg.timestamp as string,
      kind:              "frame",
      agent:             msg.from as string,
      frame_description: p?.description ?? "",
    });
  });

  // screen:approaches
  watchStream(redis, "screen:approaches", "monitor", "mon-approaches", (msg) => {
    const p = msg.payload as any;
    pushEvent({
      id:                 msg.id as string ?? crypto.randomUUID(),
      ts:                 msg.timestamp as string,
      kind:               "approach",
      agent:              msg.from as string,
      source_description: p?.source_description ?? "",
      approach_text:      p?.approach_text ?? "",
    });
  });

  // screen:results
  watchStream(redis, "screen:results", "monitor", "mon-results", (msg) => {
    const p    = msg.payload as any;
    const sc   = p?.score ?? {};
    const ev: FeedEvent = {
      id:               msg.id as string ?? crypto.randomUUID(),
      ts:               msg.timestamp as string,
      kind:             "score",
      agent:            msg.from as string,
      overall:          sc.overall ?? 0,
      raw_score:        sc.raw_score ?? 0,
      net_impact:       sc.net_impact ?? 0,
      total_items:      sc.total_items ?? 0,
      items:            p?.items ?? [],
      dimension_scores: p?.dimension_scores ?? {},
      source_description: p?.source ?? "",
    };
    pushEvent(ev);
    scoreHistory.push({ ts: ev.ts, overall: ev.overall! });
    if (scoreHistory.length > 100) scoreHistory.shift();
  });

  // agent:health
  watchStream(redis, "agent:health", "monitor", "mon-health", (msg) => {
    const agent = msg.agent as string;
    if (!agent) return;
    agentStatus[agent] = {
      last_seen:    msg.timestamp as string,
      score:        (msg.score as number) ?? 0,
      task:         (msg.what as string) ?? "",
      top_positive: (msg.top_positive as string) ?? "",
      top_negative: (msg.top_negative as string) ?? "",
      help_needed:  (msg.help_needed as string) ?? "",
      status:       "alive",
    };
  });

  // Poll stream depths every 3s
  setInterval(async () => {
    for (const stream of ["screen:frames", "screen:approaches", "screen:results", "agent:health"]) {
      try { streamDepths[stream] = await redis.xLen(stream); } catch { /* ignore */ }
    }
  }, 3000);

  // Mark stale agents
  setInterval(() => {
    const now = Date.now();
    for (const [name, info] of Object.entries(agentStatus)) {
      agentStatus[name].status = now - new Date(info.last_seen).getTime() > 60_000 ? "stale" : "alive";
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
        { COUNT: 20, BLOCK: 1000 },
      );
      if (results) {
        for (const { messages } of results) {
          for (const { id, message } of messages) {
            handler(JSON.parse(message.data));
            await redis.xAck(stream, group, id);
          }
        }
      }
    } catch { await new Promise(r => setTimeout(r, 500)); }
  }
}

// ── API routes ────────────────────────────────────────────────────

app.get("/feed",    (_req, res) => res.json({ events: feed.slice(0, 60), agents: agentStatus, stream_depths: streamDepths }));
app.get("/status",  (_req, res) => res.json({ agents: agentStatus, stream_depths: streamDepths, score_history: scoreHistory.slice(-20), avg_score: avg(), updated_at: new Date().toISOString() }));
app.get("/streams", (_req, res) => res.json({ depths: streamDepths }));

function avg() {
  const recent = scoreHistory.slice(-20);
  return recent.length ? Math.round(recent.reduce((s, e) => s + e.overall, 0) / recent.length * 100) / 100 : 0;
}

// ── HTML Dashboard ────────────────────────────────────────────────

app.get("/", (_req, res) => {
  res.setHeader("Content-Type", "text/html");
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Courier New',monospace;background:#0d1117;color:#c9d1d9;display:grid;grid-template-rows:auto 1fr;height:100vh;overflow:hidden}
header{padding:12px 20px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:16px;background:#161b22}
header h1{color:#58a6ff;font-size:15px;white-space:nowrap}
#pulse{width:8px;height:8px;border-radius:50%;background:#3fb950;animation:blink 1.5s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
#tick{color:#8b949e;font-size:11px}
.layout{display:grid;grid-template-columns:220px 1fr 220px;gap:0;overflow:hidden;height:100%}

/* Sidebar */
.sidebar{padding:12px;border-right:1px solid #21262d;overflow-y:auto;display:flex;flex-direction:column;gap:10px}
.sidebar.right{border-right:none;border-left:1px solid #21262d}
.panel{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px}
.panel h2{color:#58a6ff;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}

/* Agent cards */
.agent-card{padding:8px;border-radius:4px;border:1px solid #30363d;background:#0d1117;margin-bottom:6px}
.agent-card:last-child{margin-bottom:0}
.agent-name{font-size:12px;font-weight:bold;color:#e6edf3;margin-bottom:4px}
.agent-task{font-size:10px;color:#8b949e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px}
.agent-score{height:3px;background:#21262d;border-radius:2px;margin-bottom:4px}
.agent-score-fill{height:3px;border-radius:2px;background:#3fb950;transition:width .5s}
.badge{display:inline-block;font-size:9px;padding:1px 5px;border-radius:3px;font-weight:bold}
.alive{background:#0d4429;color:#3fb950}.stale{background:#3d0014;color:#f85149}
.struggling{color:#f0883e;font-size:9px}.ok{color:#3fb950;font-size:9px}

/* Stream depths */
.depth-row{display:flex;justify-content:space-between;font-size:11px;padding:2px 0;border-bottom:1px solid #21262d}
.depth-row:last-child{border:none}
.stream-name{color:#8b949e;font-size:10px}
.stream-count{color:#e3b341;font-weight:bold}

/* Score mini-chart */
.score-num{font-size:28px;color:#3fb950;font-weight:bold;line-height:1}
.score-trend{font-size:11px;color:#58a6ff;margin-top:2px}
.dim-row{display:flex;justify-content:space-between;align-items:center;font-size:10px;padding:2px 0}
.dim-name{color:#8b949e;width:100px;flex-shrink:0}
.dim-bar{flex:1;height:4px;background:#21262d;border-radius:2px;margin:0 6px}
.dim-fill{height:4px;border-radius:2px}
.dim-val{color:#c9d1d9;width:28px;text-align:right}

/* Feed */
.feed{overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
.event{border-radius:6px;padding:10px 12px;border-left:3px solid;font-size:12px;position:relative}
.event-frame{background:#0d1f3c;border-color:#1f6feb}
.event-approach{background:#1a0d2e;border-color:#6e40c9}
.event-score{background:#0d2318;border-color:#238636}
.event-score.low{background:#2d0c0c;border-color:#f85149}
.event-meta{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.event-type{font-size:9px;font-weight:bold;text-transform:uppercase;letter-spacing:1px}
.type-frame{color:#58a6ff}.type-approach{color:#bc8cff}.type-score{color:#3fb950}.type-score.low{color:#f85149}
.event-agent{font-size:9px;color:#8b949e}
.event-time{font-size:9px;color:#484f58;margin-left:auto}
.event-body{color:#c9d1d9;line-height:1.5}
.approach-label{font-size:9px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-top:6px;margin-bottom:2px}
.score-big{font-size:22px;font-weight:bold;color:#3fb950}
.score-big.low{color:#f85149}
.items-list{margin-top:6px;display:flex;flex-direction:column;gap:3px}
.item{font-size:10px;padding:2px 6px;border-radius:3px;display:flex;gap:6px}
.item.pos{background:#0d2318;color:#3fb950}.item.neg{background:#2d0c0c;color:#f85149}
.item-dim{font-size:9px;opacity:.6;flex-shrink:0;width:90px}
.item-obs{flex:1}
.item-impact{font-weight:bold;flex-shrink:0}
.approach-text{color:#c9d1d9;font-size:11px;max-height:80px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical}
.empty{color:#484f58;font-size:11px;text-align:center;padding:40px 0}
</style>
</head>
<body>
<header>
  <div id="pulse"></div>
  <h1>Swarm Monitor — Screen Embedding Pipeline</h1>
  <span id="tick">connecting...</span>
</header>
<div class="layout">

  <!-- Left: Agents -->
  <div class="sidebar" id="agents-sidebar">
    <div class="panel">
      <h2>Agents</h2>
      <div id="agent-cards"><p class="empty" style="padding:10px">waiting...</p></div>
    </div>
    <div class="panel">
      <h2>Pipeline</h2>
      <div style="font-size:10px;color:#8b949e;line-height:1.8">
        <div style="color:#58a6ff">frame-capture</div>
        <div style="padding-left:8px">↓ screen:frames</div>
        <div style="color:#bc8cff">embedder</div>
        <div style="padding-left:8px">↓ screen:approaches</div>
        <div style="color:#3fb950">approach-tester</div>
        <div style="padding-left:8px">↓ screen:results</div>
        <div style="color:#e3b341">monitor / refiner</div>
      </div>
    </div>
  </div>

  <!-- Center: Feed -->
  <div class="feed" id="feed">
    <p class="empty">waiting for swarm activity...</p>
  </div>

  <!-- Right: Stats -->
  <div class="sidebar right">
    <div class="panel">
      <h2>Avg Score</h2>
      <div class="score-num" id="avg-score">—</div>
      <div class="score-trend" id="score-trend"></div>
    </div>
    <div class="panel">
      <h2>Last Dimensions</h2>
      <div id="dimensions">
        ${["intent_capture","cognitive_state","specificity","noise_resistance"].map(d =>
          `<div class="dim-row"><span class="dim-name">${d.replace("_"," ")}</span><div class="dim-bar"><div class="dim-fill" id="dim-${d}" style="width:50%;background:#21262d"></div></div><span class="dim-val" id="dimv-${d}">—</span></div>`
        ).join("")}
      </div>
    </div>
    <div class="panel">
      <h2>Stream Depths</h2>
      <div id="depths"><p style="color:#484f58;font-size:11px">loading...</p></div>
    </div>
  </div>

</div>
<script>
const DIM_COLORS = {intent_capture:"#58a6ff",cognitive_state:"#bc8cff",specificity:"#3fb950",noise_resistance:"#e3b341"};
let lastEventId = null;

function reltime(ts) {
  const s = Math.round((Date.now() - new Date(ts).getTime()) / 1000);
  if (s < 5)  return "just now";
  if (s < 60) return s + "s ago";
  return Math.round(s/60) + "m ago";
}

function agentColor(name) {
  if (name.includes("capture"))  return "#58a6ff";
  if (name.includes("embed"))    return "#bc8cff";
  if (name.includes("tester"))   return "#3fb950";
  return "#8b949e";
}

function renderAgents(agents) {
  const el = document.getElementById("agent-cards");
  const entries = Object.entries(agents);
  if (!entries.length) { el.innerHTML = '<p style="color:#484f58;font-size:11px">waiting...</p>'; return; }
  el.innerHTML = entries.map(([name, a]) => {
    const pct = Math.round((a.score || 0) * 100);
    return '<div class="agent-card">' +
      '<div style="display:flex;justify-content:space-between;align-items:center">' +
        '<div class="agent-name" style="color:' + agentColor(name) + '">' + name + '</div>' +
        '<span class="badge ' + a.status + '">' + a.status + '</span>' +
      '</div>' +
      '<div class="agent-task">' + (a.task || "idle") + '</div>' +
      '<div class="agent-score"><div class="agent-score-fill" style="width:' + pct + '%"></div></div>' +
      '<div style="display:flex;justify-content:space-between">' +
        '<span class="' + (a.help_needed === "ok" ? "ok" : "struggling") + '">' + (a.help_needed || "—") + '</span>' +
        '<span style="font-size:9px;color:#484f58">' + reltime(a.last_seen) + '</span>' +
      '</div>' +
      (a.top_positive ? '<div style="font-size:9px;color:#3fb950;margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">+ ' + a.top_positive + '</div>' : '') +
      (a.top_negative ? '<div style="font-size:9px;color:#f85149;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">− ' + a.top_negative + '</div>' : '') +
    '</div>';
  }).join("");
}

function renderDepths(depths) {
  const el = document.getElementById("depths");
  const short = { "screen:frames": "frames", "screen:approaches": "approaches", "screen:results": "results", "agent:health": "health" };
  el.innerHTML = Object.entries(depths).map(([k,v]) =>
    '<div class="depth-row"><span class="stream-name">' + (short[k]||k) + '</span><span class="stream-count">' + v + '</span></div>'
  ).join("") || '<p style="color:#484f58;font-size:11px">no data</p>';
}

function renderDimensions(dimScores) {
  for (const [dim, sc] of Object.entries(dimScores || {})) {
    const pct = Math.round((sc.overall || 0.5) * 100);
    const fill = document.getElementById("dim-" + dim);
    const val  = document.getElementById("dimv-" + dim);
    if (fill) { fill.style.width = pct + "%"; fill.style.background = DIM_COLORS[dim] || "#58a6ff"; }
    if (val)  val.textContent = (sc.overall || 0).toFixed(2);
  }
}

function renderAvgScore(events) {
  const scores = events.filter(e => e.kind === "score" && e.overall !== undefined);
  if (!scores.length) return;
  const avg = scores.slice(0,10).reduce((s,e) => s + e.overall, 0) / Math.min(scores.length, 10);
  const el = document.getElementById("avg-score");
  el.textContent = avg.toFixed(2);
  el.style.color = avg >= 0.6 ? "#3fb950" : avg >= 0.45 ? "#e3b341" : "#f85149";
  const trend = scores.length > 2 && scores[0].overall > scores[scores.length-1].overall ? "↑ improving" : "→ stable";
  document.getElementById("score-trend").textContent = trend;
}

function renderEvent(ev) {
  if (ev.kind === "heartbeat") return "";

  let cls = "event event-" + ev.kind;
  let typeCls = "event-type type-" + ev.kind;
  let body = "";

  if (ev.kind === "frame") {
    body = '<div class="event-body">' + (ev.frame_description || "(no description)") + '</div>';
  }

  if (ev.kind === "approach") {
    const txt = ev.approach_text || "";
    body = '<div class="approach-label">source</div>' +
           '<div style="color:#8b949e;font-size:10px;margin-bottom:4px">' + (ev.source_description || "") + '</div>' +
           (txt
             ? '<div class="approach-label">proposed approach</div><div class="approach-text">' + txt + '</div>'
             : '<div style="color:#484f58;font-size:10px;font-style:italic">⚠ model returned empty approach</div>');
  }

  if (ev.kind === "score") {
    const isLow = (ev.overall || 0) < 0.45;
    cls += isLow ? " low" : "";
    typeCls += isLow ? " low" : "";
    const pct = Math.round((ev.overall || 0) * 100);
    body = '<div style="display:flex;align-items:baseline;gap:8px">' +
           '<span class="score-big' + (isLow ? " low" : "") + '">' + (ev.overall || 0).toFixed(2) + '</span>' +
           '<span style="font-size:10px;color:#8b949e">raw ' + (ev.raw_score||0).toFixed(1) + ' | ' + (ev.total_items||0) + ' items | net ' + (ev.net_impact||0) + '</span>' +
           '</div>';
    if (ev.source_description) {
      body += '<div style="color:#484f58;font-size:10px;margin-top:3px">' + ev.source_description + '</div>';
    }
    const items = ev.items || [];
    if (items.length) {
      const pos = items.filter(i => i.impact > 0).sort((a,b) => b.impact - a.impact).slice(0,3);
      const neg = items.filter(i => i.impact < 0).sort((a,b) => a.impact - b.impact).slice(0,3);
      body += '<div class="items-list">' +
        pos.map(i => '<div class="item pos"><span class="item-dim">' + i.dimension.replace("_"," ") + '</span><span class="item-obs">' + i.observation + '</span><span class="item-impact">+' + i.impact + '</span></div>').join("") +
        neg.map(i => '<div class="item neg"><span class="item-dim">' + i.dimension.replace("_"," ") + '</span><span class="item-obs">' + i.observation + '</span><span class="item-impact">' + i.impact + '</span></div>').join("") +
        '</div>';
    } else {
      body += '<div style="color:#484f58;font-size:10px;margin-top:4px;font-style:italic">⚠ no scored items — model may have returned malformed JSON</div>';
    }
  }

  const label = ev.kind === "frame" ? "FRAME CAPTURED" : ev.kind === "approach" ? "APPROACH PROPOSED" : "SCORED";
  const ts = ev.ts ? reltime(ev.ts) : "";

  return '<div class="' + cls + '">' +
    '<div class="event-meta">' +
      '<span class="' + typeCls + '">' + label + '</span>' +
      '<span class="event-agent" style="color:' + agentColor(ev.agent||"") + '">' + (ev.agent||"") + '</span>' +
      '<span class="event-time">' + ts + '</span>' +
    '</div>' +
    body +
  '</div>';
}

async function poll() {
  try {
    const data = await fetch('/feed').then(r => r.json());
    const events = data.events || [];

    // Feed
    const feedEl = document.getElementById("feed");
    if (events.length === 0) {
      feedEl.innerHTML = '<p class="empty">waiting for swarm activity...</p>';
    } else {
      const html = events.map(renderEvent).filter(Boolean).join("");
      if (html) feedEl.innerHTML = html;
    }

    // Sidebar
    renderAgents(data.agents || {});
    renderDepths(data.stream_depths || {});
    renderAvgScore(events);

    // Dimensions from most recent score
    const latestScore = events.find(e => e.kind === "score" && e.dimension_scores);
    if (latestScore) renderDimensions(latestScore.dimension_scores);

    document.getElementById("tick").textContent = "updated " + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById("tick").textContent = "⚠ disconnected";
    document.getElementById("pulse").style.background = "#f85149";
  }
}

poll();
setInterval(poll, 2000);
</script>
</body>
</html>`);
});

// ── Start ──────────────────────────────────────────────────────────

async function main() {
  const redis = createClient({ url: REDIS }) as RedisClientType;
  await redis.connect();
  console.log("[monitor] Connected to Redis");
  startWatching(redis);
  app.listen(PORT, () => {
    console.log(`\n🟢  Monitor running → http://localhost:${PORT}`);
    console.log(`    Feed         → http://localhost:${PORT}/feed`);
    console.log(`    Status       → http://localhost:${PORT}/status\n`);
  });
}

main().catch(console.error);
