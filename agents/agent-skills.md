# Agent skills

*Depth — packaged procedural knowledge as a standardized, composable unit that a harness can load, call, and update independently of the base model.*

**TL;DR:** A **skill** is a self-contained, structured description of *how* to do a repeatable procedure — steps, tools, expected inputs and outputs, exit conditions — packaged so an agent harness can load, invoke, and iterate on it without touching the model weights. Skills sit between raw prompts (too unstructured) and fine-tuning (too heavy) as a middle unit of behavior. SKILLER (2026) is the current strongest recipe for *automatically* generating high-quality skills for small models using language-level RL.

**Prereqs:** [_harness-optimization](_harness-optimization.md)
**Related:** [darwinx](darwinx.md), [procedural-memory](procedural-memory.md), [rlvr](../post-training/rlvr.md)

---

## What it is

Prompt engineering scales poorly when the harness has to reuse the same sub-procedure across many tasks: you either duplicate the sub-procedure into every top-level prompt or hide it in fragile Python glue. A **skill** packages that sub-procedure as a first-class object — with a name, a schema for its inputs/outputs, a description the LM can retrieve on, and a body the LM executes.

Structurally a skill looks like:

- **Name** — kebab-case identifier the router matches on.
- **Description** — one paragraph the LM reads to decide whether the skill applies.
- **Signature** — typed inputs and expected outputs.
- **Body** — the actual procedure (a prompt template, a tool-call recipe, a mini control-flow graph).
- **Exit criteria** — the verifier or heuristic that decides the skill succeeded.

A harness's *skill library* is the union of all such objects it can invoke. Optimizing an agent then becomes optimizing this library: add a new skill, rewrite a bad one, retire a dead one.

## How it works

Two orthogonal questions: **who authors skills**, and **how are they trained / kept fresh**.

**Authoring.** Manual authoring is the baseline (write skills like functions). SKILLER's contribution is **language-level RL**: run the agent, score each trajectory with a verifier, summarize the verifier feedback as a natural-language critique, and use that critique to rewrite the relevant skill. No gradient flows — the "policy update" is a diff on the skill's text.

**Composition.** At inference time, the harness retrieves a shortlist of candidate skills (by description similarity or router LM), attempts one, and can chain to another based on the result. A skill can call other skills, giving the library the shape of a program.

**Lifecycle.** Skills accumulate technical debt (the same way code does): outdated tool APIs, superseded strategies, silently-broken exit criteria. Harnesses that use skills seriously add a retirement rule — e.g. drop any skill unused in $N$ rounds *unless* it holds the record on some verifier.

## Why it matters

- **Composable**: skills swap cleanly across base models — a skill written for one 9B model works when swapped to a 4B or a 70B, because the interface is external.
- **Auditable**: an operator can read a library and see what the agent knows how to do. Weight-based knowledge is opaque; skill libraries are legible.
- **Cheap to update**: rewriting one skill is orders of magnitude cheaper than fine-tuning a whole model, and doesn't invalidate the rest of the library.
- **Small-model-friendly**: SKILLER's Qwen-9B with SKILLER-generated skills gains **+4.3 to +20.4 points** across 5 benchmarks; the 4B variant gains **+1.8 to +13.3** — and matches closed-source frontier on single-skill tasks. Skills are the current best answer for small on-device agents.

## Gotchas & tricks

- **Skill sprawl.** Without retirement, the library grows without improving success rate. Track per-skill utility (invocations × pass rate) and prune ruthlessly.
- **Router mispicks.** If two skills have overlapping descriptions, the router picks wrong. Enforce a description-uniqueness check when admitting new skills.
- **Exit-criterion drift.** A skill can silently start passing its own exit check while producing worse outputs. Cross-validate exit criteria against a held-out verifier periodically.
- **Language-level RL ≠ RL.** No gradient, no theoretical convergence guarantees. Empirically strong on small models; still an open question whether it saturates before parameter-updating RL.

## Sources

- Paper: *SKILLER: Language-Level Reinforcement Learning for Reusable Skill Extraction in Small Language Models* — Dang, Xiong, He, Li (Shanghai AI Lab / SYSU), 2026, [arXiv:2608.10538](https://arxiv.org/abs/2608.10538)
- Related concept: Anthropic's "skills" formatting for Claude — public docs, 2025, [claude.ai](https://claude.ai) — an early production-facing standardization of the skill format.
