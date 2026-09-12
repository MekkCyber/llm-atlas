# Proof-Refinement RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An RL post-training loop for natural-language mathematical proofs that replaces symbolic solvers with an LLM verifier. The model proposes a proof, an LLM critic scores rigor and points at flawed steps, the model refines, and the refined proof feeds back as the RL rollout. Applied to a Nemotron 3 Ultra checkpoint, this pipeline reached 30/42 at IMO 2026 — above the gold-medal cutoff — without a Lean/Isabelle stack.

**Prereqs:** [rlvr](../rlvr.md), [long-cot-rl](long-cot-rl.md), [prm](prm.md)
**Related:** [rejection-sampling](../rejection-sampling.md), [orm](orm.md), [../cot-reward-model](../cot-reward-model.md)

---

## What it is

Olympiad proofs cannot be formalized cheaply enough to plug into RLVR's usual solver-verified reward. Proof-Refinement RL substitutes an **LLM verifier** for the symbolic checker, and adds a **refine** step so the model can react to critique before the reward is booked. The rollout is: *propose → critique → repair → re-verify*, all in natural language.

## How it works

Given a warm-started proof-writer (SFT on curated olympiad proofs):

1. **Propose.** Sample a full proof for problem $p$: $y_0 \sim \pi_\theta(\cdot \mid p)$.
2. **Critique.** A verifier LLM $V$ scores the proof and produces localized feedback: $\big(s_0, c_0\big) \gets V(p, y_0)$, where $s_0$ is a rigor score and $c_0$ is a natural-language critique naming the weakest step.
3. **Refine.** The policy consumes the critique and produces a repaired proof: $y_1 \sim \pi_\theta(\cdot \mid p, y_0, c_0)$.
4. **Re-verify.** $\big(s_1, c_1\big) \gets V(p, y_1)$. Iterate until $s_i$ crosses an acceptance threshold or a step budget is exhausted.
5. **RL update.** Backprop through the final $s_i$ (and optionally intermediate $s_i$) with a GRPO/PPO update on the *sequence of refinements*.

The verifier is periodically retrained on human-graded proof samples to prevent reward-model drift.

## Why it matters

- **Unlocks unformalizable domains.** Olympiad geometry, combinatorics, and analysis are painful to formalize; LLM-graded rigor gives an RL signal where symbolic verification would take months of Lean engineering.
- **Refinement is the RL signal.** Framing the trajectory as propose→critique→repair means the model *learns to react to criticism*, not just to write correct proofs on the first pass. This mirrors how humans do olympiad math.
- **Open-recipe result.** 30/42 at IMO 2026, above gold-medal threshold, with fully-open weights, data, and code — first credible open counterpart to closed systems like AlphaProof.

## Gotchas & tricks

- **Verifier gaming.** The policy will find critiques the verifier consistently forgives; retrain the verifier on adversarial samples and hold out a human-graded eval slice.
- **Reward hacking through length.** LLM verifiers over-reward long, hedged proofs; pair the rigor score with a length penalty and a "conclusion-must-match-claim" gate.
- **Off-policy critique.** Sampling critiques from a strong external model (as opposed to the policy's own critic head) helps early on; switch to on-policy critique as the model matures.
- **Warm-start matters.** Direct RL from a base model diverges — an SFT pass on curated proof traces is required before the loop stabilizes.
- **Batch across refinement depth.** Trajectories of length $k$ have different variance than length-1; group by depth for a lower-variance GRPO baseline.

## Sources

- Paper: *An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics* — Moshkov, Ge, Armstrong, Du, Mahdavi, Gitman (NVIDIA), 2026 — [arXiv:2609.10712](https://arxiv.org/abs/2609.10712)
- Related: RLVR (verifiable rewards) — the symbolic-verifier ancestor this generalizes.
- Related: Process reward models — supply the localized "which step failed" signal Proof-Refinement RL relies on.
