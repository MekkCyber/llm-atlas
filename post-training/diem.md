# DIEM — Dynamic Important Example Mining
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Online prompt-level curriculum for RL fine-tuning. At each optimizer step, DIEM estimates each sample's marginal contribution to policy improvement via **gradient alignment** with the average update direction, then **reweights** the batch under a constraint that preserves gradient magnitude. Fully automated; consistently beats static and heuristic sample-selection baselines on reasoning benchmarks.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md)
**Related:** [rl-prompt-curation.md](rl-prompt-curation.md) · [rejection-sampling.md](rejection-sampling.md) · [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

Most data-centric RFT methods rank prompts *once* before training (difficulty scoring, verifier-pass-rate binning) and treat that ranking as fixed. But sample value is non-stationary: an example the policy solves at step 500 is no longer providing useful gradient at step 1500. Static curricula waste rollout budget on solved-in-place prompts and starve the policy of useful hard-but-solvable prompts.

DIEM makes the ranking dynamic: at every step, each sample's importance is re-estimated from its *current* gradient contribution.

## How it works

Two stacked components inside each optimizer step:

**1. Gradient-alignment importance estimator.** For a mini-batch $\{(q_i, o_i, r_i)\}$, compute per-sample gradient $g_i = \nabla_\theta L_i$. The marginal contribution of sample $i$ to policy improvement is approximated by its alignment with the batch's aggregate direction:

$$
\text{imp}_i \propto \langle g_i, \bar{g} \rangle \quad \text{where} \quad \bar{g} = \frac{1}{N} \sum_j g_j
$$

Cheap approximation to a leave-one-out signal; computable with a single extra dot product per sample once gradients exist.

**2. Constrained batch reweighting.** Assign each sample a weight $w_i$ maximizing aggregate importance:

$$
\max_w \sum_i w_i \cdot \text{imp}_i \quad \text{s.t.} \quad \left\| \sum_i w_i g_i \right\| = \|\bar{g}\|
$$

The magnitude constraint keeps the effective step size fixed — high-importance samples get emphasized but the optimizer's step norm doesn't blow up. Solvable in closed form under simple parameterizations.

Net effect: each step spends its gradient budget where it currently helps most, without changing the base RL algorithm (GRPO, PPO, whatever).

## Why it matters

- **Consistent wins over static and dynamic baselines** across reasoning benchmarks, per the paper's ablations.
- **Orthogonal to the RL algorithm.** Sits on top of GRPO / PPO / mirror-descent variants without modification.
- **Attacks the right waste.** RL fine-tuning is bottlenecked by rollout cost. Making each rollout count more directly reduces wall-clock to convergence.
- **A principled baseline for curriculum research.** Prior work relied on heuristics (temperature-based, verifier-pass-rate bins); DIEM's gradient-alignment framing gives a well-defined target to beat.

## Gotchas & tricks

- **Gradient-alignment approximation is noisy at small batch sizes.** The signal-to-noise ratio in $\langle g_i, \bar{g} \rangle$ improves with $N$; expect DIEM to shine at batch sizes ≥ 64 more than at 8.
- **Magnitude constraint prevents runaway upweighting** but also caps the maximum benefit from a single very-high-importance sample. If you have a strong prior (e.g., known-hard AIME problems), consider a separate injection rather than relying on DIEM to find them.
- **Interacts with GRPO's own reweighting.** GRPO already normalizes advantages within a group. DIEM operates a level up (across prompts within the batch); the two compose but the interaction hasn't been ablated separately.
- **Not a replacement for prompt curation.** DIEM decides *within* a candidate pool. Curating the pool itself (dedup, difficulty binning, rejection sampling) still matters.

## Sources

- Paper: *Dynamic Important Example Mining for Reinforcement Finetuning* — Tan, Wu, Chen, Zhao, Sun, Liu, Chang, Zhang, Sun, Wu, Xie, Qi — HKU / Tencent, 2026 — arxiv.org/abs/2608.29252.
- Code (promised): github.com/hrtan/DIEM.
