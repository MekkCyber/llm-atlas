# N-GRPO — Embedding-Level Neighbor Mixing for GRPO

*Depth — a GRPO variant that injects exploration at the embedding level by mixing each anchor token with its nearest semantic neighbors.*

**TL;DR:** Standard [GRPO](grpo.md) explores via token-level sampling, which often yields paraphrases (low diversity) rather than genuinely different solution strategies. Adding Gaussian noise to input embeddings breaks semantic consistency. **N-GRPO** sits between the two: at the input embedding for each token, mix in the embeddings of its **nearest semantic neighbors** (top-$k$ in embedding space). This keeps the input on the local semantic manifold while injecting genuine diversity into the rollout. On DeepSeek-R1-Distill-Qwen models across sizes, N-GRPO beats GRPO baselines on math reasoning and generalizes to OOD tasks.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [rl-prompt-curation.md](rl-prompt-curation.md), [rlvr.md](rlvr.md)

---

## What it is

A drop-in modification to GRPO's rollout step. The policy parameters, advantage computation, KL term, and clipped ratio loss are unchanged. The only difference is *how diversity is injected* during the $G$-way rollout per prompt.

## How it works

### The standard GRPO rollout

For prompt $q$, sample $G$ trajectories from $\pi_{\theta_{\text{old}}}$ at temperature $T > 0$. Diversity comes from temperature-sampling on the output distribution.

### The N-GRPO rollout

At each decoding step, for the *input* embedding of each token in the context (or a designated subset of positions), replace it with a convex mixture of the anchor and its $k$ nearest neighbors in embedding space:

$$
\tilde{e}(t) = \alpha \cdot e(t) + \frac{1 - \alpha}{k} \sum_{j=1}^{k} e(\text{nbr}_j(t))
$$

where $\text{nbr}_j(t)$ are the top-$k$ nearest tokens to $t$ in the embedding table (cosine or L2). $\alpha \in [0, 1]$ controls the mixing strength; $k$ is small (typical $k \in \{1, 4, 8\}$).

The mixed embedding $\tilde{e}(t)$ is what the transformer sees, so the entire rollout drifts along the *semantic manifold* of the anchor — close enough to remain meaningful, distant enough to explore genuinely alternative continuations.

### Why neighbors and not Gaussian noise

Random noise in embedding space lands the input *off* the manifold of natural tokens. Neighbor mixing stays on-manifold by construction — the neighbor set is exactly the tokens the language model considers semantically similar to the anchor. The diversity injected is therefore *language-aware*.

## Why it matters

- **Cleanly addresses GRPO's known under-exploration problem** without adding hyperparameter complexity beyond $\alpha, k$.
- **No extra training signal needed** — uses only the existing embedding table as the source of neighbors.
- **Generalizes out of distribution.** The paper reports gains on OOD reasoning tasks, suggesting the regularization is structural rather than benchmark-specific.
- **Stacks orthogonally with temperature, top-$p$, and reward-shaping tricks** — operates at a different point in the rollout pipeline.

## Gotchas & tricks

- **Tune $\alpha$ from the conservative end.** $\alpha = 1$ recovers vanilla GRPO; $\alpha = 0$ replaces every input with neighbor blends and destroys grounding. The paper's working range is around $\alpha \in [0.7, 0.95]$.
- **Don't apply to all positions.** The paper's recipe selectively mixes a subset of positions; mixing every token tends to collapse the prompt's identity.
- **Neighbor selection is computed once.** The top-$k$ for each token is a function of the embedding table, not the policy state, so it can be precomputed and reused across the entire RL run.
- **Distill-target backbones see the largest gains.** Reported on DeepSeek-R1-Distill-Qwen; less-distilled bases may need different $\alpha$.
- **Not a replacement for prompt curation.** Diversity from neighbor mixing complements diverse prompts ([rl-prompt-curation.md](rl-prompt-curation.md)) — it doesn't substitute for them.

## Sources

- Paper: *N-GRPO: Embedding-Level Neighbor Mixing for Enhanced Policy Optimization* — Zhu, Yu, Di, Zhu, Ant Group / Zhejiang U., 2026 — [arXiv:2606.10768](https://arxiv.org/abs/2606.10768).
- Related: [grpo.md](grpo.md), [rlvr.md](rlvr.md).
