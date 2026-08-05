# Agent Harnesses
*Taxonomy — the control-flow scaffolds around an LLM that turn it into a task-executing agent.*

**TL;DR:** A harness is the outer loop that feeds observations, tool results, and state back into an LLM to produce actions. Choices along three axes — **state location** (in-context vs external), **verification** (blind vs environment-audited), and **structure** (single-loop vs multi-role) — cover most of the design space. The frontier is moving toward external state and per-step environment verification.

**Related taxonomies:** [_agent-memory](_agent-memory.md)
**Depth files covered here:** *(none yet — this taxonomy anchors an empty region)*

---

## The problem

An LLM alone is a token predictor, not an agent. A harness wraps it so that:
- Outputs are parsed into typed actions (tool calls, sub-plans, state updates).
- Environment responses are re-injected as observations.
- Progress toward a goal is tracked across many LM invocations.

Naïvely, everything above lives in the LM's context — task description, prior actions, tool results, self-assessments — and grows monotonically. This causes three failure modes:

1. **Context rot.** Long contexts degrade attention; important state gets ignored.
2. **Compounding self-hallucination.** A hallucinated intermediate assessment gets reused as fact in the next step.
3. **Cost.** Each step re-tokenises the full growing context.

Every harness variant is a different way to attack one or more of these.

## The shared pattern

```
    goal + state ──▶ LLM ──▶ action ──▶ environment
        ▲                                     │
        └───── observation ◀──────────────────┘
```

The variables that vary:
- **What is in "state"?** Some subset of: task description, action history, tool results, self-generated notes, external structured state.
- **Who updates state?** The LLM (via self-assessment) or the environment (via verified facts)?
- **How many LLMs in the loop?** One in each step, or specialised roles (planner, executor, critic)?

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **ReAct** (Yao 2022) — *no depth file yet* | Interleave a "thought" with each action; single LLM, single context | Context grows monotonically; self-assessments unverified | Baseline; short-horizon tasks |
| **Reflexion** (Shinn 2023) — *no depth file yet* | Add a self-critique step after each failure; retry with reflection in context | Reflections may hallucinate; still one growing context | Modest-horizon tasks with easy retries |
| **Plan-and-Execute** — *no depth file yet* | Planner LLM writes a plan; executor LLM runs each step | Plan can go stale; poor recovery from surprises | Structured multi-step tasks |
| **Manage-Execute-Audit** (LongHorizon-Harness, 2026) — *no depth file yet* | External task-state store; Manager/Executor/Auditor triplet; state updates only on environment-verified facts | Requires environment able to verify claims | Long-horizon real-world tasks |
| **CodeAct / Agent-in-code** — *no depth file yet* | Agent's action space is Python code; environment is a REPL | Requires safe code execution; awkward for non-computational tools | Tool-heavy workflows over structured data |
| **Multi-agent role decomposition** (Camel, MetaGPT) — *no depth file yet* | Multiple specialised agents (analyst, engineer, reviewer) collaborate | High token cost; coordination overhead | Complex creative tasks with clear roles |

## How to choose

**Default for a new short-horizon agent:** ReAct with a small tool set and a hard step budget.

**Default for a long-horizon agent (>20 steps):** externalise task state (Manage-Execute-Audit-style). External state + environment-verified updates is the single biggest reliability win.

**When to add multi-role structure:** if the task decomposes cleanly into distinguishable specialties (research + write + review) *and* token budget isn't the binding constraint.

**When to avoid multi-role:** almost every other time. Extra LLM roles multiply cost and coordination bugs; a single well-scaffolded loop usually wins.

## Adjacent but distinct

- **[Agent memory](_agent-memory.md)** — how facts persist *across* interactions. Harnesses handle *within* a single task; memory handles *across* tasks/sessions.
- **[Agent skills](agent-skills.md)** — reusable procedural units the harness can invoke. Skills are components; the harness is the runtime.
- **RL post-training for agents** — how the *model* is trained. Orthogonal to harness choice; a well-trained model still needs a harness at inference.

## Sources

- Paper: *ReAct: Synergizing Reasoning and Acting in Language Models* — Yao et al., 2022.
- Paper: *Reflexion: Language Agents with Verbal Reinforcement Learning* — Shinn et al., 2023.
- Paper: *LongHorizon-Harness: Advancing Long-Horizon Agents for Real-World Tasks* — arXiv:2608.01964, 2026 — Manage-Execute-Audit pattern with external state.
- Paper: *CodeAct: Executable Code Actions Elicit Better LLM Agents* — Wang et al., 2024.
