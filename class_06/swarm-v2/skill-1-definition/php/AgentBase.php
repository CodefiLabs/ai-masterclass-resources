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
        public readonly string $whatWorked  = "",  // what's working + strongest parts
        public readonly string $whatDidnt   = "",  // what's not working + weakest parts
        public readonly string $whatMissing = "",  // what should have been included
        public readonly string $helpNeeded  = "",
        public readonly string $timestamp   = "",
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
        public readonly string $whatWorked,
        public readonly string $whatDidnt,
        public readonly string $whatMissing,
    ) {}
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
    protected string $whatWorked   = '';
    protected string $whatDidnt    = '';
    protected string $whatMissing  = '';

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
            agent:       $this->name,
            what:        $this->currentTask,
            goal:        $this->goal,
            why:         $this->reasoning,
            how:         $this->approach,
            score:       $this->lastScore,
            whatWorked:  $this->whatWorked,
            whatDidnt:   $this->whatDidnt,
            whatMissing: $this->whatMissing,
            helpNeeded:  $this->assessHelp(),
            timestamp:   now()->toIso8601String(),
        );
    }

    private function assessHelp(): string
    {
        if ($this->lastScore === 0.0) return 'No score yet — just getting started';
        if ($this->lastScore < 0.4)  return "Struggling ({$this->lastScore}) — need different approach for: {$this->whatDidnt}";
        if ($this->lastScore < 0.7)  return "Progress ({$this->lastScore}) — could improve: {$this->whatMissing}";
        return "Performing well ({$this->lastScore}) — no help needed";
    }

    // ── Memory ────────────────────────────────────────────────────

    public function addToMemory(string $task, string $result, array $score): void
    {
        $entry = new MemoryEntry(
            task:        $task,
            result:      substr($result, 0, 400),
            score:       $score['overall']      ?? 0.0,
            whatWorked:  $score['what_worked']  ?? '',
            whatDidnt:   $score['what_didnt']   ?? '',
            whatMissing: $score['what_missing'] ?? '',
        );

        $this->memory[] = $entry;
        if (count($this->memory) > $this->memoryLimit) {
            array_shift($this->memory);
        }

        // Update live state
        $this->lastScore   = $entry->score;
        $this->whatWorked  = $entry->whatWorked;
        $this->whatDidnt   = $entry->whatDidnt;
        $this->whatMissing = $entry->whatMissing;
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
            "Score: {$m->score} | Worked: {$m->whatWorked} | Didn't: {$m->whatDidnt} | Missing: {$m->whatMissing}",
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
