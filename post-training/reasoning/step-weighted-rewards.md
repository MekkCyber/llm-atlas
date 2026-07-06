# Step-weighted rewards
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reward-shaping trick for multi-step reasoning RL: when a solution fails, distribute the penalty across its steps **exponentially by step index**, so earlier invalid steps get much larger penalties than later ones. Motivated by the observation that most reasoning failures start with a bad early step whose error *cascades*. Introduced with MRPO for medical multimodal reasoning; it plugs into GRPO / RLVR pipelines with no new networks.

**Prereqs:** [grpo](../grpo.md), [prm](prm.md), [rlvr](../rlvr.md)
**Related:** [_rewards](../_rewards.md), [long-cot-rl](long-cot-rl.md), [orm](orm.md)

---

## What it is

Standard GRPO / RLVR assigns a single scalar reward per rollout, broadcast uniformly across every token in the response. For long structured reasoning traces this wastes signal: when the final answer is wrong, we don't know *which* step went wrong, and uniformly penalizing all tokens teaches the policy to change late steps as often as early ones — even though the causal error is almost always early.

Step-weighted rewards address this without training a PRM. The reward is still terminal-driven, but the *penalty* is redistributed over the $S$ steps of the trace with weights that grow exponentially toward earlier steps.

---

## How it works

For a failed rollout with steps $s_1, \ldots, s_S$, assign a per-step penalty

$$
p_i = -\alpha \cdot \exp(\gamma \cdot (S - i))
$$

so $p_1$ (the earliest step) receives the largest magnitude and $p_S$ the smallest. Then compute per-token advantages by broadcasting each step's penalty to its tokens; the rest of the update is standard GRPO clipping + KL.

MRPO (Jung et al., 2026) additionally uses a **step validity classifier** — an inexpensive checker that flags whether each step is *individually* invalid, based on task-specific structure (e.g. medical-reasoning consistency checks). Penalties are applied only to the steps flagged invalid; valid steps are left at zero. Combined with the exponential weighting, the earliest *invalid* step attracts the bulk of the gradient.

Successful rollouts are rewarded uniformly (as in normal GRPO); the asymmetry is on failure signals only.

---

## Why it matters

- **Fixes cascade errors cheaply.** MRPO reports early-stage reasoning failures dropping from **64.0% → 13.0%** on medical multimodal reasoning; outperforms much larger medical VLMs.
- **PRM-free.** No PRM training pipeline, no step-label collection, no PRM-hacking failure mode. Just a reward-shaping schedule.
- **Composable.** Slots in on top of GRPO / RLVR without algorithmic changes; the only new hyperparameters are the exponential base $\gamma$ and the scale $\alpha$.
- **Domain-general pattern.** Any long-CoT task where early parsing / perception errors dominate final failures (medicine, legal reasoning, math with figures, table QA) is a candidate.

---

## Gotchas & tricks

- **Requires a step-invalidity signal for the strongest gains.** Without a per-step validity classifier the penalty is applied to all steps of a failed trace, and the exponential weighting still helps but by less.
- **Doesn't reward good early steps.** Only penalizes bad ones on failed rollouts. If a policy consistently gets the first step right but fails later, this scheme won't help — you likely need a PRM or long-CoT length shaping instead.
- **$\gamma$ tuning.** Too small ($\gamma \approx 0$) collapses to uniform per-step rewards. Too large concentrates all gradient on step 1 and destabilizes training. MRPO's ablations settle around a moderate value that gives step 1 roughly $10\times$ the weight of step $S$ for typical $S$.
- **Step segmentation.** Same PRM caveat applies — you need a definition of what a "step" is. Structured tasks (medicine templates, math steps) work; open-ended prose doesn't.

---

## Sources

- Paper: *Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning* — Jung et al., 2026 — [arXiv:2606.31825](https://arxiv.org/abs/2606.31825).
- Related: [prm](prm.md) for step-level *learned* rewards; [long-cot-rl](long-cot-rl.md) for the uniform-reward baseline.
