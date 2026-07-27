# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distillation setup in which the *student's own trajectories* are the training states, and the *teacher* provides the per-step supervision. Unlike classical (off-policy) distillation, where student and teacher both consume fixed prompts and the student mimics the teacher's outputs on that distribution, OPD trains the student on the states the student itself would visit — closing the train/test distribution gap that plagues sequence models trained by teacher forcing.

**Prereqs:** [_post-training.md](./_post-training.md), [_rl.md](./_rl.md) (policy-gradient framing).
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [reopd.md](./reopd.md) · [long-cot-rl.md](./reasoning/long-cot-rl.md) · [_rewards.md](./_rewards.md)

---

## What it is

Given a strong teacher policy $\pi_T$ and a student $\pi_S$, OPD trains $\pi_S$ to match $\pi_T$'s next-token distribution *on states sampled from $\pi_S$*:

$$
\mathcal{L}_{\text{OPD}} = \mathbb{E}_{s \sim \pi_S}\big[ \text{KL}\big(\pi_T(\cdot\mid s)\,\|\,\pi_S(\cdot\mid s)\big) \big]
$$

The novelty is the sampling distribution: student rollouts drive the state visitation, so the student is corrected exactly where it would err at deployment. Standard SFT samples states from the teacher (or a static dataset), which leaves the student unprepared for its own mistake trajectories.

## How it works

Each iteration:

1. Roll out the student in the environment for $T$ steps to collect trajectory $\tau_S = (s_1, a_1, \dots, s_T)$.
2. For every visited state $s_t$, query the teacher for its full output distribution (or a top-$k$ approximation).
3. Update the student toward the teacher's distribution at those states.

For agentic tasks, each "step" is a full multi-turn interaction with an environment (tool call, browser action, code execution). Every rollout costs fresh environment + teacher inference, which is the dominant expense.

## Why it matters

OPD fixes the fundamental teacher-student mismatch: the student is *tested* on its own state distribution but *trained* on the teacher's. Empirically, OPD beats plain SFT and rejection-sampling SFT on agentic and reasoning tasks, especially when teacher/student capabilities are far apart or the environment is stochastic.

## Gotchas & tricks

- **Expensive.** Fully-online OPD requires teacher queries + student rollouts every step. Practical variants trade some on-policy-ness for compute — replayed prefixes ([reopd.md](./reopd.md)), stale-teacher caching, or partial rollouts.
- **Prefix trap.** If prefixes drift far from what the teacher was trained on, teacher targets become unreliable — a two-sided distribution shift. Solutions range from step-decaying sampling to reliability-aware prefix selection.
- **Teacher access.** Assumes API/model access to the teacher's *distribution*, not just samples. Some pipelines approximate this with top-$k$ log-probs.

## Sources

- Paper: *Multi-Turn On-Policy Distillation with Prefix Replay* — Liao et al., 2026 — [arXiv:2607.04763](https://arxiv.org/abs/2607.04763)
- Earlier: on-policy distillation for MT / speech has been in the KD literature since Agarwal et al. (2024) *On-Policy Distillation of Language Models*.
