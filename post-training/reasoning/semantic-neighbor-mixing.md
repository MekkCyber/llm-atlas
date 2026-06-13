# Semantic Neighbor Mixing (N-GRPO)

*Depth — a GRPO rollout-diversity trick that mixes anchor-token embeddings with their nearest semantic neighbors, injecting variation without leaving the local semantic manifold.*

**TL;DR:** GRPO's rollout phase wants diverse trajectories so the group baseline isn't degenerate, but token-level sampling produces mostly rephrased duplicates and random embedding noise corrupts meaning. **N-GRPO** dynamically constructs each input embedding by mixing the anchor token's embedding with the embeddings of its nearest neighbors in the embedding space. Diversity comes from the mix, semantic coherence comes from the local-manifold constraint. Consistent gains over strong GRPO baselines on math-reasoning benchmarks with DeepSeek-R1-Distill-Qwen.

**Prereqs:** [../grpo](../grpo.md), [long-cot-rl](long-cot-rl.md)
**Related:** [../rlvr](../rlvr.md), [../rl-prompt-curation](../rl-prompt-curation.md), [../_rl](../_rl.md)

---

## What it is

A change to how the policy's *input embeddings* are constructed during the rollout phase of GRPO. Standard GRPO samples actions from the policy's output distribution — the input embedding for each token is fixed by the tokenizer. N-GRPO instead replaces the input embedding with a **mixture** of the anchor token's embedding and the embeddings of its nearest semantic neighbors. The mixture coefficients are sampled, so different rollouts within a group see different mixed embeddings — and the resulting trajectories differ in meaningful, not merely lexical, ways.

This is positioned against two failure modes:

- **Token-level sampling diversity is shallow**: high-temperature decoding mostly produces re-orderings and paraphrases that collapse into the same reasoning step, so the group has near-identical advantages and the GRPO gradient signal becomes weak.
- **Random embedding noise corrupts semantics**: prior embedding-perturbation methods inject Gaussian noise into the input, but the noise pushes embeddings off the learned manifold, breaking semantic consistency and producing incoherent rollouts.

Neighbor mixing stays on the manifold (by construction — neighbors are *learned* nearby points) while still producing meaningfully different inputs.

## How it works

For each token position during a rollout:

1. Look up the anchor embedding $e_t$ and its $k$ nearest neighbors $\{e_{t,1}, \dots, e_{t,k}\}$ in the model's embedding table (cosine similarity).
2. Sample mixing weights $(\alpha_0, \alpha_1, \dots, \alpha_k)$ from a simplex.
3. Construct the perturbed input $\tilde e_t = \alpha_0 e_t + \sum_i \alpha_i e_{t,i}$.
4. Feed $\tilde e_t$ to the transformer instead of $e_t$.

Different rollouts in the same GRPO group sample different weights, producing semantically distinct trajectories. The rest of GRPO (group baseline, clipped ratio, KL penalty) is unchanged.

Neighbor lookup is precomputed once per token from the static embedding table — no runtime ANN cost.

## Why it matters

- **Sharper GRPO gradient signal.** When all $G$ rollouts in a group give the same advantage, the GRPO update is near-zero. Diversifying *on the input side* widens the advantage spread without changing the loss or the policy.
- **Cheap and surgical.** No new loss term, no architectural change, no extra forward passes — just a different embedding construction inside the rollout phase.
- **Drop-in to existing reasoning-RL pipelines.** Anywhere you're running GRPO over a frozen embedding table, you can swap in N-GRPO embeddings.

## Gotchas & tricks

- **Neighbor count $k$ and mixing temperature matter.** Too few neighbors → close to token-level sampling; too many → drift off the local manifold and back into the noise regime.
- **Training-only.** At inference you want the original embeddings.
- **Interacts with KL penalty.** Perturbed inputs change the reference-policy probabilities; the KL term has to be computed under the same perturbed inputs or the regularization is biased.

## Sources

- Paper: N-GRPO — Zhu et al. (2026) — [arXiv:2606.10768](https://arxiv.org/abs/2606.10768)
