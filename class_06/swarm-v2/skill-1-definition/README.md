# Skill 1: Agent Definition

An agent is a process with:
- A **role** (what it is)
- A **goal** (what it's optimizing for)
- **Memory** (what it's learned so far)
- The ability to **ask and answer the six questions**

This folder has the same agent defined in all three languages.
They all implement the same interface and can talk to each other via Redis.

## What to build here

1. Run `python/agent.py` — watch it answer the six questions
2. Run `typescript/agent.ts` — same thing, different runtime
3. Look at `php/AgentBase.php` — same logic in Laravel
4. Notice: every agent has `get_status()`, `add_to_memory()`, `build_system_prompt()`

## The core pattern

```
Role + Goal
    ↓
System prompt (injected with score memory)
    ↓
LLM call
    ↓
Answer the six questions in the response
    ↓
Store result + score in memory
    ↓
Next call gets smarter system prompt
```
