# SAGE: Structural Admissibility-Guided Exploration
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long-horizon reasoning under sparse rewards fails in two structured ways: **exploration bias** (models drift into locally plausible but structurally unstable branches) and **compounding bias** (small local deviations accumulate across depth and suppress the rare reward). SAGE fights both with additive structural priors — **algebraic sparsification** for exploration bias and **hyperbolic-space embedding** for compounding bias — layered on top of standard RL post-training.

**Prereqs:** [long-cot-rl](./long-cot-rl.md), [grpo](../grpo.md)
**Related:** [mcts](./mcts.md), [prm](./prm.md), [orm](./orm.md)

---

## What it is

The problem SAGE names: as reasoning chains get longer and rewards get sparser, RL training on LLMs stops improving and often regresses. Two mechanisms explain most of the collapse:

- **Exploration bias.** At any step, the LLM's local policy assigns high probability to branches that look locally plausible but violate global structural constraints. The rollout ends up in unrecoverable regions.
- **Compounding bias.** Every step incurs small error. In an unstructured metric space these errors add linearly (or worse), so after 50+ steps the trajectory has diverged from any correct solution.

SAGE formalizes both via **Symbolic Closure Analysis (SCA)** — a lens that characterizes how branching structure and reward sparsity interact under *local admissibility* (a step is admissible iff a locally-verifiable rule is satisfied).

## How it works

Two additive structural priors on top of a standard on-policy RL objective (typically GRPO on long CoT):

1. **Algebraic sparsification.** At each step, project the set of locally-admissible candidate branches onto **operator-indexed algebraic subspaces**. Branches outside those subspaces are downweighted. Concretely: for a reasoning task with a known set of operators (arithmetic ops, code constructs, proof steps), maintain a subspace basis and score each candidate by its component in the admitted subspace. Suppresses spurious "sounds right" branches that don't compose with the task operators.

2. **Hyperbolic structural guidance.** Embed reasoning states into a **negatively curved (hyperbolic) space**, in which distances between deep-and-nearby vs shallow-and-far states are naturally distinguished. In this geometry the density of "reachable target states" per unit distance is higher near the root and thinner at depth — supplying a **dense** depth-wise gradient signal even when rewards are sparse only at the leaves. Practically, augment the RL loss with a hyperbolic-embedding regularizer that pulls state representations toward the negatively-curved manifold.

Neither prior requires new rollouts, human labels, or a learned reward model.

## Why it matters

- **Directly attacks the long-CoT ceiling.** Reasoning RL was hitting a limit that looked structural, not reward-shaping-fixable. SAGE identifies the mechanism.
- **Transferable priors.** The two priors are additive losses; they layer on top of any RL post-training pipeline (GRPO, PPO, RLVR).
- **Big margins on the hardest tasks.** Across 12 benchmarks and 7 model families SAGE beats competitive baselines; on the Andrews-Curtis problem (an open real-world long-horizon math task) SAGE achieves an **8× improvement**.

## Gotchas & tricks

- **Operator index needs task knowledge.** Algebraic sparsification requires you to define the operator set. Works well for math/logic/code; less obvious for open-ended natural-language reasoning.
- **Hyperbolic embedding needs care.** Numerical stability near the boundary of the Poincaré ball is a known issue; use Lorentz or squared-distance formulations.
- **Loss weights matter.** Both priors are additive; too-large weights over-regularize and collapse the policy to a single admissible mode.
- **Composes with process reward models.** PRMs give per-step reward; SAGE gives per-step structural constraint. Complementary.

## Sources

- Paper: *SAGE: Mitigating Long-Horizon Reasoning Biases via Topological Guidance* — Zeng, Zhang, Yan, Zhou — Virginia Tech + Dartmouth, 2026 — https://arxiv.org/abs/2609.30192
- Related: [long-cot-rl](./long-cot-rl.md) — the RL regime SAGE operates within.
- Related: *Hyperbolic Neural Networks* — Ganea et al., 2018 — the geometric machinery SAGE reuses.
