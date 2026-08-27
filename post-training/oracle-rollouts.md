# Oracle Rollouts (OraRL)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In GRPO-style RL, each on-policy group contains few high-reward rollouts and CoT sampling is expensive. **OraRL** injects the dataset **annotation as an extra oracle rollout** into each on-policy group so the model has a guaranteed positive target, but avoids "advantage inversion" — where a high-reward oracle raises the group baseline and flips otherwise positive policy advantages — via a **decoupled advantage estimator** and **sign-balanced pruning**. Introduced by Li et al. 2026 for video MLLMs; the mechanism is model-agnostic and applies whenever supervised annotations coexist with on-policy RL.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md)
**Related:** [_rl.md](_rl.md), [_rewards.md](_rewards.md), [rejection-sampling.md](rejection-sampling.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

GRPO samples $G$ responses per prompt and uses the group's reward mean and std as the advantage baseline. When the base policy is weak, most of the $G$ rollouts are wrong — the group signal is dominated by failures and the update has little to push toward.

Datasets already contain a *known-correct* trajectory (the annotation). OraRL asks: what if we drop that annotation into the on-policy group as a `(G+1)`-th "oracle" rollout — treating it as a direct positive optimization target rather than merely a scoring reference?

The naïve version breaks: the oracle's high reward raises the group mean, and every non-oracle rollout that was above the old baseline flips below the new one. Policy rollouts that should have been rewarded now get *negative* advantages. Li et al. call this **advantage inversion** — a hazard that shows up in any hybrid on-policy + teacher/oracle scheme (SFT-warmstart with RL, teacher-forced RL, DPO-with-preferred-answers-in-batch).

## How it works

### Decoupled advantage estimator

Compute two baselines separately:

- **Oracle-free baseline** — mean and std over only the $G$ policy rollouts, ignoring the oracle. Policy advantages $A_i^{\text{policy}}$ are computed against this baseline exactly as in vanilla GRPO.
- **Directional gain + detached oracle advantage** — the oracle-policy reward gap modulates a directional term that pulls the policy toward the oracle, plus a separate detached advantage for the oracle itself.

Concretely:

$$
A_i^{\text{policy}} = \frac{r_i - \bar r_{\text{policy}}}{\sigma_{r,\text{policy}}}, \quad
A^{\text{oracle}} = g(r_{\text{oracle}} - \bar r_{\text{policy}}) \; \text{(detached, stop-grad on policy stats)}.
$$

The oracle's reward never re-enters the policy-rollout normalization, so policy advantages keep their sign.

### Sign-balanced pruning

Rather than update against all $G+1$ rollouts, keep only the oracle + the strongest positive-advantage rollout + the strongest negative-advantage rollout. This preserves the sign structure of the group while dramatically cutting the number of long-context video forward passes.

Result: OraRL takes **2.2× SFT step time**, less than half the **4.9× of GRPO+CoT**.

## Why it matters

- **Recovers useful signal from annotations.** Every RL dataset ships with annotations. Standard GRPO uses them only for reward scoring; OraRL turns them into direct optimization targets *and* keeps them from corrupting on-policy advantages.
- **Scales without CoT.** In Li et al.'s video MLLM setting, chain-of-thought sampling is the RL bottleneck (4,780 ms/decode for Video-ORA-9B with CoT vs 130 ms without). OraRL matches CoT performance without CoT, because the oracle carries the correctness signal that CoT was implicitly re-deriving.
- **Names a general hazard.** Advantage inversion is a real failure mode any time you augment on-policy groups with high-reward trajectories from another distribution (teacher, expert, past-best). Naming it makes it debuggable.

## Gotchas & tricks

- **Detach the oracle statistics.** The decoupled estimator only works if you compute the policy-rollout baseline with `stop_grad` on any oracle-derived quantity. Otherwise gradients leak through the baseline.
- **Off-policy oracles need importance correction.** Annotations were not sampled by $\pi_\theta$ — they're off-policy. Sign-gating the oracle advantage sidesteps this, but for the directional gain you're implicitly assuming the annotation is representative of "good policy behavior". This can hurt on tasks where multiple equally-valid answers exist.
- **Pruning changes the effective group size.** With sign-balanced pruning at $G=8$ down to 3 kept rollouts, variance estimates in the policy-only baseline become noisy. Use unnormalized advantages ($A = r - \bar r$) or a shrinkage estimator for very small groups.
- **Annotation quality is now a first-class knob.** Because the oracle rollout is directly optimized against, low-quality annotations are worse than absent ones. Inspect the top-loss oracle rollouts as you would inspect top-loss SFT examples.

## Sources

- Paper: *Annotations as Rollouts: Efficient and Scalable Reinforcement Learning for Video MLLMs* — Li et al., 2026 — introduces OraRL, decoupled advantage estimator, sign-balanced pruning. [arXiv:2608.20492](https://arxiv.org/abs/2608.20492).
- Related: *DeepSeekMath* — Shao et al., 2024 — GRPO baseline OraRL modifies.
