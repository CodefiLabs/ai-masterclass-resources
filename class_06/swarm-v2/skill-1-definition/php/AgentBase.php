<?php
/**
 * skill-1-definition/php/AgentBase.php
 * ──────────────────────────────────────
 * SKILL 1: Agent Definition (PHP/Laravel)
 *
 * Full implementation — not a stub.
 * Same contract as Python and TypeScript:
 * - getStatus()         → answers the six questions
 * - addToMemory()       → stores scored results
 * - buildSystemPrompt() → injects history into every LLM call
 * - act()               → calls the local model, returns response
 *
 * All inference is local via Ollama — no cloud API calls.
 *
 * Requirements:
 *   composer require guzzlehttp/guzzle
 *
 * Usage in Laravel:
 *   $agent = new EmbedderAgent();
 *   $result = $agent->act("Describe this screen frame...");
 */

namespace App\Agents;

use GuzzleHttp\Client;

// ── Value Objects ─────────────────────────────────────────────────

class AgentStatus
{
    public function __construct(
        public readonly string $agent,
        public readonly string $what,
        public readonly string $goal,
        public readonly string $why,
        public readonly string $how,
        public readonly float  $score,
        public readonly string $topPositive  = "",  // highest-impact positive observation
        public readonly string $topNegative  = "",  // highest-impact negative observation
        public readonly string $helpNeeded   = "",
        public readonly string $timestamp    = "",
    ) {}

    public function toArray(): array
    {
        return get_object_vars($this);
    }
}

class MemoryEntry
{
    public function __construct(
        public readonly string $task,
        public readonly string $result,
        public readonly float  $score,
        public readonly string $topPositive,
        public readonly string $topNegative,
        public readonly array  $impactItems = [],
    ) {}
}

// ── Two-Pass Scoring ─────────────────────────────────────────────
// Pass 1 (LLM): itemize observations with +/- impact scores
// Pass 2 (math): net_impact / √(items) → normalized 0-1 score

function deriveScore(array $items): array
{
    if (empty($items)) {
        return ['overall' => 0.5, 'net_impact' => 0, 'total_items' => 0,
                'normalized_impact' => 0.0, 'raw_score' => 50.0];
    }
    $netImpact = array_sum(array_column($items, 'impact'));
    $totalItems = count($items);
    $normalizedImpact = $netImpact / sqrt($totalItems);
    $rawScore = max(0, min(100, 50 + ($normalizedImpact * 8.0)));
    $overall = round($rawScore / 100, 3);
    return [
        'overall'            => $overall,
        'net_impact'         => $netImpact,
        'total_items'        => $totalItems,
        'normalized_impact'  => round($normalizedImpact, 3),
        'raw_score'          => round($rawScore, 1),
    ];
}

// ── Base Agent ────────────────────────────────────────────────────

abstract class AgentBase
{
    private Client $http;
    private string $ollamaModel;

    /** @var MemoryEntry[] */
    private array $memory      = [];
    private int   $memoryLimit = 5;

    // Live state (answers to six questions)
    protected string $currentTask  = 'idle';
    protected string $reasoning    = '';
    protected string $approach     = '';
    protected float  $lastScore    = 0.0;
    protected string $topPositive  = '';
    protected string $topNegative  = '';

    public function __construct(
        protected string $name,
        protected string $role,
        protected string $goal,
    ) {
        $ollamaUrl = env('OLLAMA_BASE_URL', 'http://localhost:11434');
        $this->ollamaModel = env('OLLAMA_MODEL', 'qwen3.5:latest');

        $this->http = new Client([
            'base_uri' => $ollamaUrl,
            'headers'  => [
                'content-type' => 'application/json',
            ],
        ]);
    }

    // ── Six Questions ─────────────────────────────────────────────

    public function getStatus(): AgentStatus
    {
        return new AgentStatus(
            agent:        $this->name,
            what:         $this->currentTask,
            goal:         $this->goal,
            why:          $this->reasoning,
            how:          $this->approach,
            score:        $this->lastScore,
            topPositive:  $this->topPositive,
            topNegative:  $this->topNegative,
            helpNeeded:   $this->assessHelp(),
            timestamp:    now()->toIso8601String(),
        );
    }

    private function assessHelp(): string
    {
        if ($this->lastScore === 0.0) return 'No score yet — just getting started';
        if ($this->lastScore < 0.4)  return "Struggling ({$this->lastScore}) — top issue: {$this->topNegative}";
        if ($this->lastScore < 0.7)  return "Progress ({$this->lastScore}) — working on: {$this->topNegative}";
        return "Performing well ({$this->lastScore}) — no help needed";
    }

    // ── Memory ────────────────────────────────────────────────────

    public function addToMemory(string $task, string $result, array $score): void
    {
        $items = $score['items'] ?? [];
        $derived = $score['derived'] ?? deriveScore($items);

        // Extract top positive/negative from impact items
        $positives = array_filter($items, fn($i) => $i['impact'] > 0);
        $negatives = array_filter($items, fn($i) => $i['impact'] < 0);
        usort($positives, fn($a, $b) => $b['impact'] <=> $a['impact']);
        usort($negatives, fn($a, $b) => $a['impact'] <=> $b['impact']);
        $topPos = !empty($positives) ? reset($positives)['observation'] : '';
        $topNeg = !empty($negatives) ? reset($negatives)['observation'] : '';

        $entry = new MemoryEntry(
            task:         $task,
            result:       substr($result, 0, 400),
            score:        $derived['overall'] ?? 0.5,
            topPositive:  $topPos,
            topNegative:  $topNeg,
            impactItems:  $items,
        );

        $this->memory[] = $entry;
        if (count($this->memory) > $this->memoryLimit) {
            array_shift($this->memory);
        }

        // Update live state
        $this->lastScore   = $entry->score;
        $this->topPositive = $entry->topPositive;
        $this->topNegative = $entry->topNegative;
    }

    public function buildSystemPrompt(): string
    {
        $base = implode("\n", [
            "You are {$this->name}, a {$this->role}.",
            "Goal: {$this->goal}",
            "",
            "After completing any task, structure your response as:",
            "WHAT: [what you did]",
            "HOW: [how you approached it]",
            "RESULT: [the output]",
            "CONFIDENCE: [0.0-1.0]",
            "HELP: [what would make your next attempt better]",
        ]);

        if (empty($this->memory)) {
            return $base;
        }

        $recent = array_slice($this->memory, -3);
        $history = implode("\n\n", array_map(fn(MemoryEntry $m) =>
            "Task: " . substr($m->task, 0, 80) . "\n" .
            "Score: {$m->score} | Best: {$m->topPositive} | Worst: {$m->topNegative}",
            $recent
        ));

        return "{$base}\n\nYour recent performance — use this to improve:\n{$history}";
    }

    // ── LLM ──────────────────────────────────────────────────────

    public function ask(string $prompt, ?string $imageB64 = null): string
    {
        $messages = [];
        $messages[] = ['role' => 'system', 'content' => $this->buildSystemPrompt()];

        $userMsg = ['role' => 'user', 'content' => $prompt];
        if ($imageB64) {
            $userMsg['images'] = [$imageB64];
        }
        $messages[] = $userMsg;

        $response = $this->http->post('/api/chat', [
            'json' => [
                'model'    => $this->ollamaModel,
                'messages' => $messages,
                'stream'   => false,
                'options'  => ['num_predict' => 1024],
            ],
        ]);

        $body = json_decode($response->getBody()->getContents(), true);
        return $body['message']['content'] ?? '';
    }

    public function act(string $task, ?string $imageB64 = null): string
    {
        $this->currentTask = $task;
        return $this->ask($task, $imageB64);
    }
}
