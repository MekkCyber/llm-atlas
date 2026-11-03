# Training-Inference Mismatch
*Depth — the sampling-vs-gradient gap in disaggregated LLM RL, and the DLC fix.*

**TL;DR:** Modern RL post-training stacks sample rollouts with a fast inference engine (vLLM, SGLang) and compute gradients with a separate trainer, so the two distributions never fully agree. The gap is **two** cumulative errors: a near-constant **numerical discrepancy** from the engine's kernels and precision, and a growing **policy lag** from stale weights that trainer updates amplify in a feedback loop. Importance weighting controls the lag only while effective sample size (ESS) stays above ½. **DLC** (Discrepancy-Lag Correction) centres the numerical term, clips the lag ratio, and triggers a sampler↔trainer sync when ESS drops — keeping stale-sampler deviation inside ~2× the baseline noise floor.

**Prereqs:** [_rl.md](_rl.md), [ppo.md](ppo.md), [grpo.md](grpo.md)
**Related:** [rlvr.md](rlvr.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [../systems/ray.md](../systems/ray.md) · [_post-training.md](_post-training.md)

---

## What it is

Classical on-policy RL assumes the policy that produces a rollout is the same policy that gets its gradient. In LLM RL, "same policy" is only approximately true for two independent reasons:

1. **Engine ≠ trainer.** vLLM/SGLang use different kernels, different precision (often FP16 or BF16 vs the trainer's FP32 master weights), and different scheduling. The log-prob the engine emits for a token is close to, but not equal to, what the trainer computes for the same (prefix, token) pair.
2. **Sampler weights are stale.** Rollouts are generated with weights that are $k$ gradient steps behind the current trainer weights. As training progresses, this gap grows, and trainer updates can amplify it in a feedback loop.

The paper splits the mismatch into these two terms so each can be handled with the right tool.

## How it works

### Decomposition

Define the inference-engine policy $\pi^{\mathrm{eng}}_\theta$ (what the engine actually samples from) and the training policy $\pi^{\mathrm{tr}}_\theta$ (what the trainer evaluates). Then

$$
\log \pi^{\mathrm{eng}}_\theta(a\mid s) - \log \pi^{\mathrm{tr}}_\theta(a\mid s)\;\approx\;\underbrace{\delta_{\mathrm{num}}(s,a)}_{\text{numerical}} \;+\; \underbrace{\Delta_k(s,a)}_{\text{policy lag}}
$$

- $\delta_{\mathrm{num}}$ has **approximately constant mean** across training; it's the engine's static offset.
- $\Delta_k$ **grows** with $k$ (gradient steps since sync) and is itself driven by the trainer's own updates.

### Discrepancy-Lag Correction (DLC)

Three ingredients, all cheap:

1. **Centre the numerical term.** Subtract a running mean estimate of $\delta_{\mathrm{num}}$ from the log-ratio before using it. Zeros out the static engine bias without needing to match engines.
2. **Clip the lag ratio.** Standard importance-weight clip (as in PPO) applied *to the lag component after centring*, not the raw log-ratio — so clipping budget isn't wasted on $\delta_{\mathrm{num}}$.
3. **ESS-triggered resync.** Track the effective sample size of the importance weights across the current rollout batch. When ESS falls below ½ (the paper's threshold), force a weight sync between trainer and sampler before continuing. This caps lag growth before the feedback loop takes over.

The ½ threshold comes from a formal result: unbiasedness of the IS estimator breaks down when ESS drops below ½ of batch size, after which the variance of the policy gradient grows super-linearly.

### Where it plugs in

DLC lives in the PPO/GRPO ratio computation step: instead of `ratio = exp(logp_train - logp_sampler)`, you compute the two corrections and then let the clipping / advantage weighting proceed as before.

## Why it matters

- **Silent correctness bug source.** Open-source RLVR stacks (OpenRLHF, veRL, verifiers) have shipped with variants of this mismatch for months; measured deviations can be orders of magnitude above the stated noise floor on long runs.
- **Runtime-measurable trigger.** ESS is cheap to compute per batch — no tuning, no schedule guesswork. "Sync when ESS < ½" is a drop-in rule.
- **Explains why "sync every N steps" works and doesn't.** Fixed-N sync regimes work by accident when $N$ happens to be small enough for the lag's current regime; they fail silently when the policy enters a region where $\Delta_k$ grows faster.

## Gotchas & tricks

- **Centring must track across runs, not within a batch.** $\delta_{\mathrm{num}}$ varies with model size, engine version, kernel selection. A fresh run needs a short warmup to estimate it; checkpoints should save the running mean.
- **ESS½ is a floor, not a target.** The paper reports stability at ESS½; aggressive setups can push it, but gradient variance climbs quickly below ESS¼.
- **Clip budget leakage.** If you clip the raw log-ratio (not the lag component), $\delta_{\mathrm{num}}$ steals clip budget on *every* token. DLC's centring step is why this ordering matters.
- **Interacts with partial rollouts.** Partial-rollout schemes (see [partial-rollouts.md](../systems/partial-rollouts.md)) extend the per-trajectory lag; expect ESS to drop faster and syncs to be more frequent.
- **Only validated on small models.** Qwen3-0.6B / 1.7B. Behavior on frontier-scale MoEs is open.

## Sources

- Paper: *On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning* — anonymous, 2026 — arXiv 2610.40945. Correspondence emails suggest George Washington + Duke.
- Related: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 — the importance-weighted ratio that DLC plugs into.
- Related: *DeepSeekMath (GRPO)* — Shao et al., 2024 — the group-relative version where this mismatch bites in open implementations.
