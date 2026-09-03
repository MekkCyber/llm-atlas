# SLERP Model Merging
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** **Spherical linear interpolation** of two models' weights: instead of linearly averaging parameter vectors, interpolate along the *sphere* through them, preserving the angular relationship. In post-training, this is a lightweight way to merge **separately trained experts** (each RL-tuned on one objective) into one model without joint fine-tuning. The T-Tech corporate LLM recipe uses two-stage SLERP to combine three GRPO experts — instruction following, function-calling, and internal task distribution — into a single served model that beats a ~7× larger baseline.

**Prereqs:** [grpo.md](grpo.md)
**Related:** [_post-training.md](_post-training.md), [../pre-training/model-souping.md](../pre-training/model-souping.md)

---

## What it is

Given two model weight vectors $\theta_A, \theta_B$ (typically fine-tunes from a shared base), **linear interpolation** gives $\theta = (1-t)\theta_A + t\theta_B$. **SLERP** treats the weight vectors as points on a hypersphere and interpolates along the great-circle arc:

$$
\theta = \frac{\sin((1-t)\Omega)}{\sin \Omega}\, \theta_A + \frac{\sin(t\Omega)}{\sin \Omega}\, \theta_B
$$

where $\Omega$ is the angle between $\theta_A$ and $\theta_B$ (computed as $\arccos(\hat\theta_A \cdot \hat\theta_B)$ on normalized weights). $t \in [0, 1]$ picks the mixing point along the arc.

When $\Omega$ is small (nearly co-linear vectors), SLERP $\approx$ linear interpolation. When $\Omega$ is larger, SLERP preserves magnitude and angular structure better than a straight average — important for RL-tuned models where the fine-tune direction, not just position, carries the specialization.

## How it works

### Standard two-model SLERP

Applied layer-wise (or block-wise) to two models sharing a base:

1. Compute the angle $\Omega$ between the two parameter vectors.
2. Pick a mixing weight $t$.
3. Emit the merged model per the SLERP formula.

Layer-wise application usually works better than a single global $\Omega$, since different layers rotate different amounts during fine-tuning.

### Two-stage SLERP for expert merging (T-Tech recipe)

For $N > 2$ experts, SLERP is repeated hierarchically:

- **Stage 1:** Merge experts in pairs — e.g., merge instruction-following + function-calling → intermediate model $M_1$.
- **Stage 2:** SLERP the intermediate model with the remaining expert(s) → final model.

The stage order matters: pairing more-compatible experts first (small $\Omega$) reduces cumulative angular loss.

### Why not just average?

Straight averaging of RL-tuned experts causes **cross-domain reward interference**: the average moves off the SFT-manifold in a way that neither expert's optimization anticipated. SLERP stays closer to the sphere the experts were trained on, so each expert's specialization survives better.

## Why it matters

- **Composes RL-tuned experts without joint training.** Joint multi-objective RL suffers from reward interference (semantic collapse, verbosity hacking, over-calling); training separate experts and merging with SLERP sidesteps this.
- **Deployment-simple.** No new training pass — just weight arithmetic on existing checkpoints.
- **Empirically strong at scale.** T-Tech's SLERP-merged model beats a **~7× larger** baseline on their internal Arena (69.6 vs 65.8) and absorbs 50% of platform traffic (116M req/mo) at a fraction of the fleet cost.
- **Complementary to souping.** Model souping averages same-objective checkpoints; SLERP merges *different-objective* experts. Both live in the same "post-training weight arithmetic" toolkit.

## Gotchas & tricks

- **Requires a shared base.** SLERP between models with different initializations or vocabularies is meaningless. Use it only among fine-tunes of the same base.
- **Layer-wise > global.** A single global $\Omega$ underfits the per-layer geometry. Compute $\Omega$ per layer (or per block).
- **Pair-order in multi-expert SLERP.** Merge closer experts first; leave the outlier for the last stage.
- **$t$ is not universally 0.5.** Uneven expert quality means the middle isn't optimal. Sweep $t$ against a held-out mix that stratifies to the intended deployment traffic.
- **Not a substitute for reward design.** SLERP only helps if each expert's reward is well-designed in isolation. It doesn't fix bad rewards.
- **Beware embedding drift.** If any expert re-tuned the embedding matrix aggressively, SLERPing embeddings can distort token semantics. Some recipes freeze embeddings during the per-expert RL for this reason.

## Sources

- Paper: *From Production Traffic to Post-Training: Building a Self-Hosted LLM That Covers the Corporate Request Mix* — Tsymboi et al., 2026 — [arXiv:2609.01572](https://arxiv.org/abs/2609.01572) — deployment recipe using two-stage SLERP over GRPO experts.
- Reference: SLERP originates in computer-graphics quaternion interpolation (Shoemake, 1985) and has been used for model merging since 2023 open-source recipes.
