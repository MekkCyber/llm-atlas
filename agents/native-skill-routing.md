# Native Skill Routing (Gavel)
*Depth — routing over huge skill libraries using probes on a frozen LLM's own hidden states.*

**TL;DR:** Agent harnesses usually route among skills either by dumping all skill metadata into context (which caps library size and disperses attention) or by external retrieval pipelines (which move the decision outside the agent's own capability). Gavel (Chen et al., 2026) shows the frozen LLM already carries the routing signal in its forward passes and reads it out with two linear maps and per-skill banks built at installation.

**Prereqs:** [../interpretability/README.md](../interpretability/README.md), [../architectures/_moe.md](../architectures/_moe.md)
**Related:** [modular-harness](modular-harness.md)

---

## What it is

A skill-router for agent systems that keeps skill text *out* of the model's context. The router is two small trained linear maps and a compact set of per-skill "banks" precomputed at install time; the base LLM is frozen. Given a task, Gavel produces a shortlist of skills ("glance") and then a final scoring ("verdict") without ever asking the base model to read any skill's metadata.

## How it works

Two stages, each reusing the base model's own computation.

**Glance.** For each installed skill, one forward pass at install time captures its mid-layer hidden states and reduces them into a compact per-skill bank (a small matrix). Two trained linear maps `M_task` and `M_skill` project the current task's mid-layer state and the skill banks into a shared space. Cosine (or dot-product) scores against every skill bank produce a full-library ranking in one matrix multiplication — no per-skill forward pass required. The top-K enter the shortlist.

**Verdict.** For each of the K shortlisted skills, resume the base model's forward pass with the skill's short description appended, and read out two signals: (1) the log-likelihood of a "yes, use this skill" completion, (2) an explicit yes/no judgment token. Fuse with the glance score as a *product of experts*.

Only the two linear maps are trained. Training uses task-skill pairs (positive) and hard negatives.

## Why it matters

Context-based routing caps at hundreds of skills before attention degrades and per-token cost balloons. Retrieval pipelines add a separate model (1.2B–16B in Chen et al.'s baselines). Gavel does both at zero additional parameters beyond two matrices, and — because the routing signal is in the base model — routing accuracy scales with the base model. On Qwen3-32B, it beats retrieve-and-rerank baselines by up to 13.4 points on written tasks and 21.9 points when a skill is needed mid-rollout, and beats "far larger frontier models running in Codex" on the paper's Skill-Use benchmark.

## Gotchas & tricks

- **The two linear maps are trained once.** They transfer zero-shot to new libraries at install time; only the banks change.
- **Bank staleness.** If the model is fine-tuned, banks and maps must be regenerated — they encode the *pretrained* internal geometry.
- **Product-of-experts fusion beats sum-of-experts.** Multiplying glance and verdict scores punishes disagreement; summing lets one dominate.
- **Skill descriptions still matter for the verdict.** A skill with a terse or misleading description hurts verdict accuracy even though it never entered context during glance.
- **Latency profile.** Glance is O(library size × d) as a single matmul. Verdict is O(K) forward passes on the shortlist. Tune K to your latency budget.

## Sources

- Paper: *The Router Within: Eliciting Native Skill Routing from a Frozen LLM* — Chen et al., 2026 — [arXiv:2609.15982](https://arxiv.org/abs/2609.15982).
