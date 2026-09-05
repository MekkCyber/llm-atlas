# RLVR entrance-entropy collapse
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RLVR lifts pass@1 but shrinks the policy's *solution space*, and test-time scaling (pass@k) stops paying off. Zhou & Li (2026) localize the shrinkage: nearly all likelihood mass shifts happen at *early decision tokens* — the "which strategy to try" tokens — while later computation tokens are approximately unchanged. Alternative solutions remain executable but are no longer initiated. A simple parameter-interpolation intervention between the RLVR-trained and base policies partially restores diversity.

**Prereqs:** [../rlvr.md](../rlvr.md), [_rl](../_rl.md)
**Related:** [long-cot-rl](long-cot-rl.md), [../../pre-training/model-souping.md](../../pre-training/model-souping.md), [orm](orm.md)

---

## What it is

Two observations that seem in tension:

- **RLVR pass@1 goes up.** The policy solves the average problem more reliably.
- **RLVR pass@k stops improving with k.** Test-time scaling (draw more samples, take any-correct) hits a lower ceiling than the pre-RL policy did.

Zhou & Li name the mechanism: **entrance-entropy collapse**. The policy's uncertainty about *which strategy to attempt* concentrates onto one branch during RLVR. Once that branch is chosen, subsequent computation tokens play out approximately as they always did. Solution-space shrinkage is thus a *sampling* problem, not a *knowledge* problem — the alternatives are still latent, just no longer initiated.

## How it works

### Where the likelihood mass moves

Compare the pre-RLVR policy $\pi_0$ to the post-RLVR policy $\pi_{\text{RL}}$ over trajectories sampled from $\pi_0$. Decompose the KL divergence per token position. On Countdown across two model configurations, the accumulated KL is:

- Concentrated at the **first ~5% of tokens** (the entrance region).
- Near-zero on the remaining ~95% (the computation region).

The "entrance region" corresponds to the choice of subgoal, formula, or reasoning template. RLVR sharpens this choice; downstream token distributions barely move once the entrance token is fixed.

### Solution coverage drop

**Solution coverage** (fraction of ground-truth-different solutions the policy can *initiate* under stochastic sampling) drops by **up to 67%** after RLVR — this is the pass@k ceiling in raw form.

### Parameter interpolation as a repair

Interpolate between the base and RL-trained weights:

$$
\theta_\lambda = (1 - \lambda) \theta_0 + \lambda \theta_{\text{RL}}
$$

Small $\lambda < 1$ restores entrance-region entropy (by pulling logits back toward the base) while preserving most of RLVR's pass@1 gain. This is [model-souping](../../pre-training/model-souping.md)-adjacent: same operation, different motivation.

## Why it matters

Names a concrete failure mode of RLVR and gives it a diagnosis + intervention. The community has debated whether RLVR "adds capability" or "sharpens selection"; this paper says: *sharpens selection at token 0–5%, doesn't change capability*. That has downstream consequences:

- **Pass@k is the wrong headline metric** for evaluating whether RLVR added anything new — it measures entrance diversity, not capability.
- **Staged training pipelines that keep some SFT/DPO pressure after RLVR** partially prevent the collapse — the pattern isn't inevitable.
- **The intervention is nearly free.** Weight interpolation costs one linear combination, not another training run.

## Gotchas & tricks

- **$\lambda$ is task-dependent.** Small $\lambda$ (say 0.7) works for math; different regimes may need different values. Sweep on a held-out set.
- **Not the same as adding entropy bonus during RL.** Entropy bonus during training changes the objective; interpolation is a post-hoc fix that leaves training untouched.
- **Only helps when the base policy had the solutions.** For truly-new capabilities acquired during RL (rare), interpolation loses them.
- **The 5% figure is the studied setup, not a law.** Different reasoning benchmarks and different base models place the entrance region at different token counts; measure it before assuming.

## Sources

- Paper: *Locked at the Entrance, Open Inside: Where RLVR Narrows the Solution Space* — Zhou & Li, University of Aberdeen, 2026 — [arXiv:2608.29188](https://arxiv.org/abs/2608.29188).
- Related: model souping (Wortsman et al., 2022); test-time-scaling analyses in DeepSeek-R1 and Kimi k1.5.
