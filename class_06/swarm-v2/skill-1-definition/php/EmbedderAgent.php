<?php
/**
 * skill-1-definition/php/EmbedderAgent.php
 * ──────────────────────────────────────────
 * A real, runnable PHP agent — not a stub.
 *
 * This is the PHP implementation of the same EmbedderAgent
 * that exists in Python (skill-2-orchestration/run_swarm.py).
 *
 * It:
 *   - Reads from screen:frames (Redis Stream)
 *   - Proposes an embedding approach using Claude
 *   - Reads the current refined approach from Redis (from the refiner)
 *   - Publishes results to screen:approaches
 *   - Answers all six questions via getStatus()
 *
 * Run as a Laravel queue worker:
 *   php artisan queue:work --queue=embedder
 *
 * Or standalone:
 *   php EmbedderAgent.php
 */

namespace App\Agents;

use Predis\Client as Redis;

require_once __DIR__ . '/AgentBase.php';

class EmbedderAgent extends AgentBase
{
    private Redis $redis;

    public function __construct()
    {
        parent::__construct(
            name: 'embedder-php',
            role: 'Screen Embedding Strategist',
            goal: 'Produce embedding descriptions that best capture human work intent, cognitive state, and work stage',
        );
        $this->redis = new Redis(env('REDIS_URL', 'redis://localhost:6379'));
    }

    // ── Main run loop ─────────────────────────────────────────────

    public function run(): void
    {
        $this->ensureGroup('screen:frames', 'embedders');
        echo "[{$this->name}] Listening on screen:frames\n";

        while (true) {
            $messages = $this->redis->xreadgroup(
                'embedders',
                $this->name,
                ['screen:frames' => '>'],
                'COUNT', 1,
                'BLOCK', 2000,
            );

            if (!$messages) {
                continue;
            }

            foreach ($messages as $stream => $entries) {
                foreach ($entries as $id => $data) {
                    $message = json_decode($data['data'], true);
                    $this->handleFrame($message);
                    $this->redis->xack($stream, 'embedders', $id);
                }
            }
        }
    }

    // ── Handle one frame ──────────────────────────────────────────

    private function handleFrame(array $message): void
    {
        $description = $message['payload']['description'] ?? '';
        $this->currentTask = "Proposing embedding for: " . substr($description, 0, 50) . "...";
        $this->reasoning   = 'Produce best embedding approach for this frame';
        $this->approach    = 'LLM with memory injection + current refined approach';

        // Read current refined approach from refiner (Skill 5)
        $refinedApproach = $this->redis->get('swarm:current_approach');
        $approachContext = $refinedApproach
            ? "\n\nCurrent refined approach from the research loop:\n{$refinedApproach}"
            : '';

        $prompt = <<<PROMPT
Frame description: "{$description}"

Propose the best way to embed this frame for a model learning human work behavior.
{$approachContext}

APPROACH: [describe your embedding strategy]
EMBEDDING_TEXT: [the actual text you would embed — max 150 words]
REASONING: [why this approach will produce better vector representations]
CONFIDENCE: [0.0-1.0]
PROMPT;

        $result = $this->ask($prompt);

        // Publish to screen:approaches
        $this->publish('screen:approaches', 'approach-tester', 'approach_proposal', [
            'source_description' => $description,
            'approach_text'      => $result,
            'agent'              => $this->name,
        ]);

        echo "[{$this->name}] Approach published\n";
    }

    // ── Receive score feedback (called by scorer) ─────────────────

    public function receiveScore(array $score): void
    {
        $this->addToMemory(
            task:   $this->currentTask,
            result: 'approach published',
            score:  $score,
        );

        $status = $this->getStatus();
        echo "[{$this->name}] Score received: {$status->score} | {$status->helpNeeded}\n";
    }

    // ── Redis helpers ─────────────────────────────────────────────

    private function ensureGroup(string $stream, string $group): void
    {
        try {
            $this->redis->xgroup('CREATE', $stream, $group, '0', 'MKSTREAM');
        } catch (\Exception $e) {
            // Group already exists
        }
    }

    private function publish(string $stream, string $to, string $type, array $payload): void
    {
        $msg = [
            'id'        => 'msg_' . bin2hex(random_bytes(4)),
            'from'      => $this->name,
            'to'        => $to,
            'type'      => $type,
            'payload'   => $payload,
            'context'   => $this->getStatus()->toArray(),
            'timestamp' => now()->toIso8601String(),
        ];

        $this->redis->xadd($stream, '*', ['data' => json_encode($msg)]);
        $this->heartbeat();
    }
}


// ── PHP Swarm Controller (reads scores for API) ───────────────────
// This goes in app/Http/Controllers/SwarmController.php
// Keeping it here for the class exercise — move it in real Laravel

class SwarmStatusReader
{
    private Redis $redis;

    public function __construct()
    {
        $this->redis = new Redis(env('REDIS_URL', 'redis://localhost:6379'));
    }

    public function getStatus(): array
    {
        $agents = $this->readAgentHealth();
        $scores = $this->readRecentScores(20);
        $avg    = count($scores)
            ? array_sum(array_column($scores, 'overall')) / count($scores)
            : 0;

        // Regression check — same logic as skill-4-eval/eval.py
        $trend = $this->detectTrend($scores);

        return [
            'agents'            => $agents,
            'score_history'     => $scores,
            'avg_score'         => round($avg, 2),
            'trend'             => $trend,
            'current_approach'  => $this->redis->get('swarm:current_approach') ?? 'not set',
            'stream_depths'     => $this->getStreamDepths(),
            'updated_at'        => now()->toIso8601String(),
        ];
    }

    private function readAgentHealth(): array
    {
        $messages = $this->redis->xrevrange('agent:health', '+', '-', 'COUNT', 50);
        $agents   = [];
        foreach ($messages as $id => $data) {
            $payload = json_decode($data['data'], true);
            $name    = $payload['agent'] ?? 'unknown';
            if (!isset($agents[$name])) {
                $agents[$name] = [
                    'last_seen'   => $payload['timestamp'] ?? '',
                    'score'       => $payload['score'] ?? 0,
                    'task'        => $payload['what'] ?? '',
                    'what_worked' => $payload['what_worked'] ?? '',
                    'help_needed' => $payload['help_needed'] ?? '',
                ];
            }
        }
        return $agents;
    }

    private function readRecentScores(int $limit): array
    {
        $messages = $this->redis->xrevrange('screen:results', '+', '-', 'COUNT', $limit);
        $scores   = [];
        foreach ($messages as $id => $data) {
            $payload  = json_decode($data['data'], true);
            $score    = $payload['payload']['score'] ?? [];
            $scores[] = [
                'ts'      => $payload['timestamp'] ?? '',
                'overall' => $score['overall'] ?? 0,
                'worked'  => $score['what_worked'] ?? '',
                'didnt'   => $score['what_didnt'] ?? '',
            ];
        }
        return array_reverse($scores);
    }

    private function getStreamDepths(): array
    {
        $streams = ['screen:frames', 'screen:approaches', 'screen:results', 'agent:health'];
        $depths  = [];
        foreach ($streams as $stream) {
            try {
                $depths[$stream] = $this->redis->xlen($stream);
            } catch (\Exception $e) {
                $depths[$stream] = 0;
            }
        }
        return $depths;
    }

    private function detectTrend(array $scores): string
    {
        if (count($scores) < 3) return 'insufficient data';
        $half  = intdiv(count($scores), 2);
        $early = array_slice($scores, 0, $half);
        $late  = array_slice($scores, $half);
        $avgE  = array_sum(array_column($early, 'overall')) / count($early);
        $avgL  = array_sum(array_column($late,  'overall')) / count($late);
        $delta = $avgL - $avgE;
        if ($delta > 0.05)  return 'improving ↑';
        if ($delta < -0.05) return 'declining ↓';
        return 'stable →';
    }
}
