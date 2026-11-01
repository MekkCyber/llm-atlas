# Unified Training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reframe pretraining, mid-training, SFT, and RL as *one* maximum-likelihood objective over different data sources (static corpora, labeled data, and reward-weighted model rollouts). Mix all four in a single trainer with schedulable mixing weights, rather than running three sequential stages. Reported gains: better accuracy–compute tradeoff, less forgetting, broader task coverage, and a stronger starting point for continual learning.

**Prereqs:** [_post-training.md](_post-training.md), [_rl.md](_rl.md), [ppo.md](ppo.md)
**Related:** [../pre-training/mid-training.md](../pre-training/mid-training.md), [rejection-sampling.md](rejection-sampling.md), [grpo.md](grpo.md)

---

## What it is

The standard pipeline — pretrain → mid-train → SFT → RL — treats each stage as a different algorithm. The paper's reformulation is that all four are maximum-likelihood over *different data distributions*:

- **Pretraining:** MLE over web data.
- **Mid-training:** MLE over curated / domain data.
- **SFT:** MLE over human demonstrations.
- **RL:** MLE over model rollouts weighted by scalar reward (reward-weighted regression).

The unified objective is a weighted sum of these likelihoods with time-varying weights $\lambda_i(t)$. Choosing specific $\lambda$ schedules recovers each classical pipeline as a special case.

## How it works

Let $\mathcal{D}_i$ be data source $i$ (web, curated, demonstrations, on-policy rollouts) and $w_i(x,y)$ its per-example weight (identity for text; $\exp(R/\beta)$ for RL rollouts, i.e. reward-weighted regression). The objective:

$$
L(\theta) = -\sum_i \lambda_i(t)\;\mathbb{E}_{(x,y)\sim\mathcal{D}_i}\!\left[w_i(x,y)\,\log\pi_\theta(y\mid x)\right]
$$

RL rollouts are produced *during* training, interleaved with batches drawn from the static sources. KL-to-reference regularization enters through either an explicit KL term or by keeping the reference distribution in the mixture at low weight.

Classical pipeline recovery:
- $(\lambda_{\text{web}}, 0, 0, 0)$ for a long prefix, then switch — gives standard pretrain → SFT.
- Smoothly interpolate $\lambda$'s — gives the unified recipe the paper advocates.

The practical implication is architectural: one trainer that pulls batches from each $\mathcal{D}_i$ according to the current $\lambda$, no stage-wise checkpoint handoffs.

## Why it matters

The stage-wise pipeline accumulates *forgetting* at every transition: each stage trains on data the previous never saw and drifts away from earlier behavior. Unified training replaces hard transitions with smooth mixing, which the paper shows improves both final capability and continual-learning starting points. If this holds up at frontier scale, it changes the operational shape of post-training.

## Gotchas & tricks

- **RL as reward-weighted regression** is the lossy equivalence: the $w_i = \exp(R/\beta)$ reduction is exact only in expectation for KL-regularized RL. PPO/GRPO clipping is a separate stabilization layer you may still want.
- **Infrastructure cost.** A unified trainer needs both static-data pipes *and* a rollout engine running continuously, not sequentially — more concurrent systems, harder scheduling.
- **$\lambda$ schedule tuning is the new axis.** Discrete stages had a few boundary hyperparameters; unified training has a continuous $\lambda(t)$ curve per source. Expect new ablation surface area.
- **Not the same as "do SFT + RL in alternating epochs."** The claim is simultaneous MLE over all sources; alternation is one special case of $\lambda$ scheduling.

## Sources

- Paper: *Training Foundation Models with a Single Algorithm* — Song, Zhang, Tajwar, Arora, Schneider, Khashabi, Salakhutdinov, Bagnell, Singh, Zanette — CMU / JHU / Aurora, 2026 — [arXiv:2610.40564](https://arxiv.org/abs/2610.40564).
- Reference: *Reward-Weighted Regression* — Peters & Schaal, 2007 — the RL-as-MLE reduction they lean on.
- Reference: *InstructGPT* — Ouyang et al., 2022 — the stage-wise pipeline being generalized.
