# OPSFT — On-Policy Direction-Constrained SFT
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** SFT and on-policy RL differ not just in loss but in the *geometry* of their parameter trajectories: on-policy RL continuously rotates the update direction across steps, while SFT keeps a nearly fixed direction. OPSFT precomputes on-policy directions from a small GRPO rollout budget and projects each SFT update onto that direction cone — matching on-policy generalization at >50% less training time. Introduced in *On-Policy Parameter Update Direction Underlies Generalization in LLM Post-Training* (2026, arXiv:2609.36659).

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md), [fine-tuning/README.md](fine-tuning/README.md)
**Related:** [_post-training.md](_post-training.md), [ppo.md](ppo.md)

---

## What it is

A hybrid between SFT and on-policy RL that reuses on-policy *update directions* without paying for on-policy *rollouts* at every step. The paper's premise: generalization-under-distribution-shift in post-training is predicted by how diverse the parameter-update directions are across training steps. SFT fails this test (its directions are nearly collinear); on-policy GRPO passes it; OPSFT engineers SFT to pass it too.

## How it works

Two phases:

1. **Direction extraction.** Run a short GRPO rollout burst over a representative prompt mix. For each step, record $\hat{d}_t = \nabla L / \|\nabla L\|$ — the unit-norm gradient. Collect these $\hat{d}_t$ into a *direction bank* $D$.
2. **Constrained SFT.** During the main SFT run, project each gradient $g$ onto the span of $D$ (or the top-$k$ principal directions of $D$):
   $$
   g_{\text{OPSFT}} = \mathrm{Proj}_{\mathrm{span}(D)}(g)
   $$
   Apply the standard Adam update with $g_{\text{OPSFT}}$ in place of $g$.

The direction bank is refreshed infrequently (every N epochs, with a cheap on-policy mini-burst) so OPSFT doesn't converge back to pure SFT.

## Why it matters

Reframes the SFT-vs-RL comparison as a *parameter-trajectory* story rather than a loss-function story. If direction diversity is the right unit of analysis, the field's intuition that "RL explores more" has a concrete, measurable mechanism — and a cheap imitation (OPSFT) that captures most of it without full rollouts.

Reported gains: OPSFT matches on-policy GRPO baselines on math/reasoning benchmarks with >50% wall-clock reduction, and beats pure SFT on OOD generalization while retaining SFT's efficient use of high-quality demonstrations.

## Gotchas & tricks

- **The direction bank is specific to its base model and prompt mix.** Reuse across models or very different task mixes gives up the benefit.
- **Refresh cadence matters.** Too stale → OPSFT drifts into pure SFT; too fresh → cost approaches full on-policy RL.
- **Projecting onto too few directions kills diversity.** The paper uses the full span of $D$, not top-$k$ PCA. Dimensionality reduction on $D$ was explored and hurt generalization.
- **Not a drop-in SFT replacement for all tasks.** The direction bank is derived from on-policy trajectories, so OPSFT inherits the RL-friendly task distribution. For narrow memorization tasks, pure SFT is still simpler and competitive.

## Sources

- Paper: *On-Policy Parameter Update Direction Underlies Generalization in LLM Post-Training* — Hou, Sun, Zhang et al., 2026 — [arXiv:2609.36659](https://arxiv.org/abs/2609.36659).
