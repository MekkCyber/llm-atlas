# Reward-Model Oversensitivity
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** "More accurate reward model ⇒ better policy" is wrong. Many popular RMs are *oversensitive* — they assign different continuous scores to equally good responses, and that variance biases RL. The paper reframes RM quality on two axes (**discriminative ability** vs **specificity**) and proposes a training-free Monte-Carlo-dropout algorithm that bins continuous rewards into a small set of discrete clusters.

**Prereqs:** [_rewards.md](_rewards.md), [_rl.md](_rl.md)
**Related:** [cot-reward-model.md](cot-reward-model.md), [reasoning/orm.md](reasoning/orm.md), [reasoning/prm.md](reasoning/prm.md)

---

## What it is

Standard RM evaluation reports *pairwise accuracy*: on held-out preferences `(A > B)`, does the RM assign `r(A) > r(B)`? The paper shows that a theoretically perfect-accuracy RM can still be *oversensitive*, assigning materially different scalar scores to responses that should be interchangeable. Downstream this biases policy gradients toward whichever member of an equivalence class happens to score higher, which can produce bad policies.

Two diagnostic axes:

- **Discriminative ability** — can the RM tell *different-quality* responses apart? (≈ classical accuracy)
- **Specificity** — does it *avoid* discriminating *equal-quality* responses? (the missing axis)

A well-behaved RM has high values on both. Pairwise accuracy measures only the first.

## How it works

- For any neural RM, enable dropout at inference time (Monte-Carlo dropout).
- Score each candidate response **K times** with stochastic forward passes; estimate the mean and uncertainty of the reward.
- Cluster the continuous scores into a small number of discrete reward bins: responses whose distributions overlap within the uncertainty band collapse into the same bin.
- Use the **bin index** as the RL reward instead of the raw continuous score. Equivalent responses now produce identical reward, eliminating the oversensitivity bias.
- Training-free: works on any neural RM with dropout layers, no fine-tuning.

## Why it matters

- Reframes RM evaluation: the field has been optimizing pairwise accuracy and ignoring an orthogonal failure mode that directly hurts downstream policies.
- Discretization is **model-agnostic** and pluggable on top of existing RMs (ORMs, PRMs, CoT RMs, preference RMs).
- The "discriminative + specificity" pair correlates with downstream policy quality where pairwise accuracy doesn't — a better evaluation rubric.

## Gotchas & tricks

- Number of bins is a tuning knob: too few collapses the signal, too many recovers the oversensitivity. Paper recommends 3–5 for typical preference RMs.
- Requires dropout layers in the RM at inference; some served RMs have dropout fused out by inference engines — needs a config change.
- MC sampling cost scales linearly in K (≈8 in the paper). For high-throughput serving this is non-trivial; amortize by caching.

## Sources

- Paper: *Discretizing Reward Models* — Viswanathan, Wang, Hazarika, Nagpal, Wu, Neubig, Mao, CMU / Meta Superintelligence Labs, 2026 — arXiv:2606.21795.
