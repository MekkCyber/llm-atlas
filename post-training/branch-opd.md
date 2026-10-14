# BranchOPD — Branching On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation only sees one continuation per decision; BranchOPD identifies the *uncertain* decision points (student entropy + teacher–student disagreement), resamples several student continuations from a shared prefix at those points, and applies token-level teacher supervision to every branch. Shared-prefix reuse cuts student-generated tokens 19–31% versus independent rollouts while covering many more trajectories per training step. Reported +18.55 pp mean@16 math accuracy and +13.8 pp ALFWorld success vs vanilla OPD.

**Prereqs:** [on-policy-distillation](../post-training/on-policy-distillation.md), [grpo](../post-training/grpo.md)
**Related:** [rejection-sampling](../post-training/rejection-sampling.md), [pr-opd](../post-training/pr-opd.md), [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

An on-policy distillation variant that branches at *informative* decision points inside a student rollout, re-samples several continuations from each branch, and distills the teacher onto all of them. Operates at token-level decisions in math and at within-turn decisions in agent tasks.

## How it works

1. **Run a main rollout** of the student policy.
2. **Score every decision** by `α · H(student) + β · KL(student‖teacher)` — i.e. where is the student most uncertain *and* most disagreeing with the teacher.
3. **Pick top-$k$ branch points** (budget-aware).
4. **At each branch point**, sample $n$ additional student continuations sharing the same prefix.
5. **Apply token-level OPD** loss on the main trajectory and every branch.
6. **Protocol-boundary repair (PBR):** at tool-call boundaries (agent tasks), apply targeted supervision to curb repetition and wasted generations.

Shared prefixes mean the student only generates the *delta* for each branch, not full new rollouts.

## Why it matters

In OPD, information lives at a few decision points per rollout — most tokens are low-entropy filler. BranchOPD concentrates supervision on exactly those pivot points, where the student most needs guidance. For agentic post-training, where a single rollout can be very long and expensive, branching turns one expensive trajectory into many cheap ones.

Reported results vs vanilla OPD:

- Math: **+18.55 pp** avg@16 accuracy.
- ALFWorld: **+13.8 pp** mean ID/OOD success.
- Beats Multi-Rollout OPD (n=7) across every evaluated benchmark.
- 19% fewer student-generated tokens on ALFWorld, 31% fewer on WebShop vs seven independent trajectories.

## Gotchas & tricks

- The scoring function is deliberately *disagreement*-aware, not just entropy-aware. Random branch selection or always-branch-at-turn-start underperforms; the paper ablates this.
- $k$ (branch points) and $n$ (continuations per branch) interact strongly — more branches ≠ better once the per-step budget is spent.
- PBR is specifically for agent protocols where the student otherwise learns to repeat tool invocations; math-only workloads don't need it.
- Shared-prefix caching needs careful attention mask bookkeeping — the branches share KV cache up to the branch point.

## Sources

- Paper: *Branching On-Policy Distillation* — anon, 2026 — [arXiv:2610.13481](https://arxiv.org/abs/2610.13481)
