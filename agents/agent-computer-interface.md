# Agent-Computer Interface (ACI)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An **ACI** is the typed, guarded, state-preserving action surface an LLM agent uses to drive a computer domain — the analog of a good REPL, purpose-built for a language model. Introduced by SWE-agent (2024) for software engineering; the pattern generalises to any long-horizon domain where raw shell commands are too fragile for an LLM to drive reliably. **AutoTrainess (2026)** applies it to LLM post-training: typed operations for planning, data-prep, launch, evaluation, and logging, with state preserved across long runs.

**Prereqs:** [README](README.md)
**Related:** [../post-training/README.md](../post-training/README.md), [../systems/ray.md](../systems/ray.md)

---

## What it is

Give an LLM agent shell access and it *can* do most tasks, but each command is a chance to (a) mistype, (b) leak internal state (chdir drift, env-var pollution), or (c) do something silently destructive. An ACI replaces the raw shell surface with a small vocabulary of **typed operations** that:

- Only allow **valid state transitions** for the domain (open a file → edit a range, not "raw sed").
- **Return structured feedback** (typed success / error, not just exit codes and captured stdout).
- **Preserve state** for the agent to inspect later (working directory, open files, active experiment).
- Enforce **preconditions** (e.g., can't launch a training job without a validated config; can't merge without a passing eval).

The design lesson from SWE-agent: *"the agent's mistakes come from the shape of its tools, not the size of its brain."* Narrower, typed, feedback-rich tools reduce error rates more than a larger model does.

## How it works

An ACI is a set of tools exposed via function-calling / MCP with these design invariants:

1. **Typed inputs and outputs** — parameter schemas, structured error objects with codes + hints.
2. **Guarded state transitions** — reject invalid state moves at the interface layer; return an actionable error, not a stack trace.
3. **Persistent state view** — a "world model" the agent can read back at any point (open files, running jobs, checkpoint history), not just the last command's stdout.
4. **Preview / dry-run modes** — for destructive operations, a preview surface (show the diff / show the config that would be launched) before commit.
5. **Domain-specific richness** — operations that match the domain's actual verbs (edit-lines, launch-training, eval-checkpoint), not generic shell primitives.

### AutoTrainess mapping (post-training domain)

Typed operations in AutoTrainess cover the post-training lifecycle:

- **Plan** — describe an iteration (what benchmark, what data, what recipe); the ACI validates against a schema.
- **Data preparation** — build a benchmark-aligned dataset; the ACI records provenance and passes a decontamination check.
- **Launch** — start a training job with a validated config; ACI attaches the state to a persistent experiment record.
- **Evaluate** — score a checkpoint; ACI records score + config for later comparison.
- **Log** — narrative log of decisions across iterations, so a resume is not "start over."

## Why it matters

- **Reliability of long-horizon agent runs.** Failure rates on long agent tasks are dominated by *interface friction*, not model reasoning. ACIs that surface typed errors and preserve state improve success rates without touching the model.
- **Composable environments.** SWE-agent's ACI generalises to browsers (BrowserGym), OS-level (OSWorld), and now training loops (AutoTrainess). The pattern is domain-agnostic; the vocabulary is not.
- **Auditable behaviour.** Structured operations + persistent state make agent runs replayable and reviewable — a prerequisite for putting agents on real production systems.

## Gotchas & tricks

- Too narrow an ACI hurts. If the typed vocabulary doesn't cover what the domain needs, the agent escapes into raw shell and you've lost the guarantees.
- Error messages are prompts. The error format the ACI emits is what the model sees next — treat it as design surface, not diagnostics.
- State preservation is not caching. The ACI's persistent view must reflect *actual* system state, not a stale mirror; otherwise the agent's mental model diverges from reality and it makes worse decisions.
- Dry-run / preview surfaces are underused. For any destructive op (launch, delete, deploy), a preview reduces high-cost failures dramatically.
- ACIs are per-domain. There is no universal ACI; the point is that each long-horizon domain gets its own well-shaped one.

## Sources

- Paper: *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering* — Yang et al., 2024 — introduces the ACI framing for software-engineering agents.
- Paper: *AutoTrainess: Teaching Language Models to Improve Language Models Autonomously* — Yu et al., 2026 — [arXiv:2606.31551](https://arxiv.org/abs/2606.31551) — ACI applied to LLM post-training.
