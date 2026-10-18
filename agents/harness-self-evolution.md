# Harness Self-Evolution
*Depth — framework for agents that revise their own surrounding software.*

**TL;DR:** An agent's "harness" is everything around the model: prompts, memory, workflows, tools, repo-level control logic, plugin composition. Harness self-evolution is the class of methods that let the agent revise those components from feedback — with changes persisting across runs — while model weights stay frozen. This paper formalizes the problem, separates the update *target* from the update *mechanism*, gives continual-PAC bounds on generalization/retention, and proposes an admission rule based on fresh validation data.

**Prereqs:** [README.md](README.md), [safety/README.md](../safety/README.md)
**Related:** [rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

A vocabulary and a safety story for a class of systems that already exist in the wild (self-rewriting prompts, agentic coding tools that modify their own repo, meta-prompting loops). The framework separates:

- **Update target.** What can change: interpreted state (prompts, memory) vs. tool/code/plugin rewrites. The latter has much bigger blast radius.
- **Update mechanism.** How changes are proposed (evolutionary search, meta-prompting, self-reflection) and admitted (test suite, held-out validation).

## How it works

1. Define a validation distribution `D_val` whose samples arrive over time, independent of the agent's own actions.
2. Each candidate update `u` is scored on a *fresh* draw from `D_val`.
3. Admit `u` only if its empirical improvement exceeds a continual-PAC-derived threshold with high probability.
4. Changes propagate; the admission rule is re-evaluated at each round against a new validation draw. The paper derives bounds on cumulative generalization/retention under this rule.

## Why it matters

Self-modifying harnesses are proliferating without a shared vocabulary or safety story. The explicit "what can change, under what admission rule" split lets reviewers audit a self-evolving system without reading its full change log, and the PAC bound gives an actual knob to tune per blast-radius.

## Gotchas & tricks

- "Fresh validation data" is the critical assumption. If the agent influences how new validation samples are drawn (e.g. by changing which users use it), bounds break — treat this as a boundary condition, not a solved problem.
- Tool/code rewrites have blast radius beyond the harness (filesystem, external services). The admission rule says *keep* or *roll back*; it doesn't say *contain*. Independent safeguards are still needed.
- Framework-level — not a specific algorithm. Compatible with evolutionary search, meta-prompting, self-rewrite loops; pick the proposer based on the target.

## Sources

- Paper: *Self-Evolving Agent Harness* — Wei et al., Zhongguancun Academy / USTC / Zhongguancun Institute of AI, 2026 — [arXiv:2610.18813](https://arxiv.org/abs/2610.18813)
