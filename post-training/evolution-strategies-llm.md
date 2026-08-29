# Evolution Strategies for LLM Reasoning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Evolution Strategies (ES) is a **gradient-free, weight-perturbation** post-training method — instead of sampling tokens and backpropagating a policy gradient, sample small random weight perturbations, score each with a rollout, and update the parameters toward the perturbations that scored best. Applied to LLM reasoning, ES yields **broader reasoning coverage** than GRPO (more distinct reasoning modes, higher pass@k) by preserving policy diversity across training. Diagnoses GRPO's entropy collapse and parameter drift as the mechanistic reasons its coverage narrows; ES's weight-space perturbation is orthogonal to token-space rollouts and doesn't fall into the same trap.

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md)
**Related:** [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [rlvr.md](rlvr.md)

---

## What it is

Policy-gradient methods (PPO, GRPO) explore in *action space*: for each prompt, sample multiple response *rollouts* and update weights toward the ones with higher reward. Diversity comes from the policy's own sampling stochasticity — and it collapses as training sharpens the policy.

ES explores in *weight space*: sample N small perturbations `θ + σ·ε_i` (ε_i ∼ N(0, I)), evaluate each perturbed policy on a batch of tasks, and update `θ` toward the perturbations that scored best. No backpropagation through the policy is required; the "gradient" is estimated from the population's reward-weighted average of the perturbations.

## How it works

Each ES step:

1. **Sample perturbations.** Draw N noise vectors `ε_i ∼ N(0, I)` and form `θ_i = θ + σ · ε_i`.
2. **Evaluate.** For each `θ_i`, run the perturbed policy on a batch of reasoning prompts with a verifiable reward. Get `r_i`.
3. **Rank/normalize rewards.** Convert `r_i` to a centered score `s_i` (rank-based or z-scored).
4. **Update.** `θ ← θ + (α / (N·σ)) · Σ_i s_i · ε_i`.

Compared to GRPO:
- **Exploration operator** — coherent, whole-policy weight perturbations vs per-token sampling stochasticity.
- **Diversity dynamics** — different perturbations are different behavioral hypotheses; keeping many alive is intrinsic. Policy-gradient collapses onto the highest-mean-reward mode.
- **Memory profile** — ES needs N forward passes but *no backward*, so memory per rollout is small. Compute is the trade.

## Why it matters

- **Broader reasoning coverage.** On reasoning benchmarks the paper studies, ES yields higher pass@k across k > 1 than GRPO at comparable compute — real diversity, not just noise.
- **Mechanistic account of GRPO's narrowing.** The paper attributes GRPO's coverage loss to entropy collapse and correlated parameter drift; ES sidesteps both because perturbations are decorrelated and evaluated independently.
- **Complementary, not necessarily replacement.** ES may not beat GRPO on pass@1; the useful frame is a coverage-vs-sharpening trade with regimes for each.
- **First serious ES-vs-GRPO study for reasoning.** RL-for-reasoning has been mono-cultured on GRPO for two years; even if the default doesn't change, the axis matters.

## Gotchas & tricks

- **N is a rollout multiplier.** N=64–256 typical; each perturbation needs enough rollouts on enough prompts to give a meaningful reward, so real-cost is `N · rollouts_per_perturbation`.
- **σ tuning matters.** Too small → all perturbations behave the same, no signal. Too large → perturbations wander out of the useful basin. Anneal σ down over training.
- **Rank-based normalization is safer than z-score.** Rank-based `s_i` is scale-invariant and robust to reward outliers; z-scoring blows up under bimodal reward distributions.
- **Full-precision perturbations are wasteful for LLMs.** Practical implementations perturb only a subset of parameters (LoRA-shaped, or attention Q/K only) — the paper discusses which subsets preserve the coverage benefit.
- **Doesn't get you the same tokens.** Even at matched pass@1, ES-trained and GRPO-trained models produce different reasoning traces — style-sensitive downstream tasks (formatting, brevity) may prefer one or the other.

## Sources

- Paper: *Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO* — Ba et al. (Huawei Noah's Ark / SUSTech / CityU HK), 2026 — arXiv:2608.27351.
