# Pivot-SD — Pivot Self-Distillation for Masked Diffusion LMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Masked diffusion LMs (dLMs) have a credit-assignment problem: a few early mask commits sharply collapse the posterior over remaining positions, so most of a response is shaped by those pivots. Pivot-SD picks pivots by *information gain* over still-masked positions and supervises only them — cross-entropy for pivots on correct trajectories, targeted unlikelihood for pivots on failed trajectories. With only 200 questions × 4 rollouts, it beats both full-sequence SFT and budget-matched diffusion RL on LLaDA-8B. Introduced in *Pivot-SD* (2026, arXiv:2610.03665).

**Prereqs:** [_post-training.md](_post-training.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [grpo.md](grpo.md), [rlvr.md](rlvr.md), [reasoning/length-penalty.md](reasoning/length-penalty.md)

---

## What it is

For an autoregressive LM, each token is one commit; credit assignment is linear. For a masked dLM, each denoising step commits a block of masks jointly, and the *entropy drop* over the remaining masks varies wildly — some commits do almost nothing, a few resolve most of the posterior. Pivot-SD only supervises the high-impact commits.

## How it works

For each training example $(x, y)$ and sampled trajectory $\tau$:

1. **Pivot selection.** For each commit step $t$ in $\tau$, measure the information gain:
   $$
   \mathrm{IG}_t = H(\text{masks before } t) - H(\text{masks after } t)
   $$
   Keep the top-$k$ commits by $\mathrm{IG}_t$ as the pivots of $\tau$.
2. **Pivot supervision.**
   - If $\tau$ is **correct** (reaches the right answer): standard cross-entropy on each pivot's committed tokens.
   - If $\tau$ is **failed**: targeted unlikelihood on the pivot's committed tokens — only the pivots, not the full failed trajectory, which keeps already-locked-in text from being punished unnecessarily.
3. Non-pivot commits in both correct and failed trajectories receive no supervision.

Trajectory correctness is checked by the same rule-based verifier used for RLVR.

## Why it matters

Masked dLMs are an open frontier for post-training. Most existing recipes (SFT, RL) inherit autoregressive assumptions and either train on the final text (losing the denoising structure) or assign rewards to whole denoising steps (missing which commit mattered). Pivot-SD is dLM-native: the "which commit mattered" question is answered mechanically by information gain, not by learned attribution.

Reported results: on LLaDA-8B-Instruct with only 200 questions × 4 rollouts, Pivot-SD beats both full-sequence SFT and budget-matched diffusion RL across math and code benchmarks.

## Gotchas & tricks

- **The $k$ in top-$k$ is the main hyperparameter.** Too large → almost everything is a pivot, which collapses to full-sequence SFT. Too small → signal is too sparse.
- **Unlikelihood on pivots only is important.** Full-trajectory unlikelihood punishes text that was locked in by *earlier* decisions than the one that failed; punishing it hurts transfer.
- **Information gain is measured against the student's own uncertainty.** Teacher-pivot selection (using a teacher dLM's posterior) was tried and gave similar quality, so student-side selection is the simpler default.
- **Not applicable to autoregressive LMs.** The whole trick depends on parallel masked commits having differing information content. For AR, every token has similar structure.

## Sources

- Paper: *Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models* — Kim, Hong, Choi et al., 2026 — [arXiv:2610.03665](https://arxiv.org/abs/2610.03665).
