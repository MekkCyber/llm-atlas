# Agent Harness Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of retraining the LLM, iteratively update the *harness* around it — system prompts, tool interfaces, output schemas, control-flow logic — from execution feedback. A harness optimizer runs the agent on training scenarios, diagnoses failures, proposes harness edits, and keeps the ones that help. The current frontier adds a learned *curriculum* over scenarios so that the training mix evolves alongside the harness (ActiveSaddler, 2026).

**Prereqs:** [README.md](README.md) (agents overview)
**Related:** [belief-state-management.md](belief-state-management.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

An "agent" is a tuple of (frozen LLM, prompts, tool schemas, control code, memory policy). Harness optimization keeps the LLM frozen and iterates on everything else. Signals come from executing the agent on training scenarios: pass/fail, trace, cost, reward. Updates are LLM-authored prompt edits, schema tweaks, or control-code patches, applied to the next iteration.

Harness optimizers (DSPy's optimizers, GEPA, text-grad-style systems, ActiveSaddler) differ in:

- **What they edit.** Prompts only, schemas, control code, or all three.
- **How they diagnose failures.** Metric regression, structured error taxonomies, abstracted "failure-pattern arms."
- **How they choose scenarios.** Fixed order, random, or (ActiveSaddler) a learned curriculum.

## How it works

Standard loop:

1. Run the current harness on a batch of scenarios; record traces and rewards.
2. Diagnose: assign each failure a *pattern* (missing tool call, wrong format, silent swallow, loop).
3. Propose an edit (LLM-authored) targeting one or more patterns.
4. A/B test the proposal; keep the edit if the aggregate metric improves.

ActiveSaddler's addition is step 0: *which scenarios to use for this batch?* It models the curriculum as a non-stationary bandit over failure-pattern arms. Each arm has an estimated "learning progress" — expected improvement if the optimizer re-targets that pattern. The sampler adaptively balances revisiting known weaknesses against exploring new ones, and the set of discovered patterns itself grows over time.

## Why it matters

- **Keeps model weights frozen.** The strongest agents are often API-only models you cannot fine-tune. Harness optimization is the main lever available.
- **Fast iteration.** No GPUs required beyond inference; each round is minutes, not days.
- **Transferable artifacts.** Prompts and schemas port across LLM backbones with modest edits; a learned LoRA does not.
- **ActiveSaddler's gains are concrete.** +4.4 pass@1 on GAIA2 and +7.5 on Terminal-Bench 2.0 over the same optimizer with a fixed scenario order.

## Gotchas & tricks

- **Metric gaming.** Prompt edits that improve the eval set but hurt held-out scenarios are common; always keep a disjoint validation pool.
- **Pattern taxonomy quality bounds the method.** If "failure" is a single category, the curriculum collapses to uniform sampling. Investing in a structured taxonomy pays back fast.
- **Non-stationarity bites.** A fixed scenario order quickly becomes stale as the harness evolves — the curriculum problem is not a bug, it's the point.
- **Overfitting to the optimizer's own LLM.** When the LLM that *proposes* edits is the same as the one running the agent, the harness drifts toward self-flattering patterns. Mixing proposer backbones helps.

## Sources

- Paper: *ActiveSaddler: Automated Curriculum Learning for Agent Harness Optimization* — Park, Kim, Zhang, Han, et al., Microsoft / KAIST / POSTECH, 2026 — [arXiv:2610.00906](https://arxiv.org/abs/2610.00906).
- Related: DSPy optimizers (Khattab et al.), GEPA, text-grad — earlier harness-optimization frameworks without a learned curriculum.
