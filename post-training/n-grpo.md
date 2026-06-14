# N-GRPO (Embedding-Neighbor Mixing)
*Depth — GRPO variant that injects rollout diversity at the embedding level by mixing each anchor token with its nearest semantic neighbors.*

**TL;DR:** GRPO needs diverse rollouts to estimate good advantages, but token-level sampling tends to redo rephrasings of the same trajectory and random-noise embedding perturbations break semantics. **N-GRPO** perturbs at the embedding level using each anchor token's *nearest semantic neighbors* — bounded, on-manifold mixing that yields diverse rollouts without garbage. Consistent gains over GRPO on math reasoning across DeepSeek-R1-Distill-Qwen sizes.

**Prereqs:** [grpo.md](grpo.md)
**Related:** [_rl.md](_rl.md) · [rejection-sampling.md](rejection-sampling.md)

---

## What it is

A drop-in modification of GRPO's rollout step. Instead of sampling tokens autoregressively at temperature T, N-GRPO substitutes an *embedding-level* perturbation: at each position, the input embedding for the next-step computation is a convex mixture of the anchor token's embedding and its k-nearest neighbors in embedding space.

The downstream RL update is standard GRPO — same clipped ratio, same group-relative advantage, same KL term. Only the rollout distribution changes.

---

## How it works

### Anchor token selection

The standard policy still chooses the next token at each step (autoregressive sample at temperature T). That token is the *anchor*.

### Neighbor mixing

For each anchor token, retrieve its k nearest neighbors in the model's embedding table (cosine similarity or learned metric). Form a mixture:

```
e_mixed = (1 - α) · e_anchor + α · Σ w_i · e_neighbor_i
```

with mixing weight `α` small (typically <0.1) and weights `w_i` softmax'd by similarity. The mixed embedding feeds into the next step's attention.

### Why on-manifold

The neighbor set is restricted to *embedding-space* neighbors, which by construction live on the local semantic manifold. Random-noise perturbations leave the manifold and produce out-of-distribution rollouts. Token-level resampling stays on the manifold but oversamples paraphrases of the same trajectory.

### GRPO update

The G rollouts (now diverse at the embedding level) feed standard GRPO with group-mean baseline.

---

## Why it matters

- **Resolves the diversity-vs-validity tradeoff in GRPO rollouts.** Diversity matters most when rewards are sparse and most rollouts collapse to the same wrong answer; on-manifold perturbation is exactly the right knob.
- **No backbone changes.** Slot into existing GRPO pipelines without retraining the embedding table.
- **Empirically robust.** Consistent gains across multiple model sizes and OOD math benchmarks.

---

## Gotchas & tricks

- **k and α are coupled.** Too few neighbors with high α reproduces noise-like perturbation; many neighbors with low α reduces to the original embedding. Sweep both.
- **Embedding similarity is metric-dependent.** Cosine works for most pretrained tokenizers; for models with normalized embeddings, dot product is equivalent. Validate the neighbor list looks semantically sensible.
- **Gradient routing.** The neighbor lookup is non-differentiable; treat the mixing as a fixed augmentation per rollout, not as a trainable component (otherwise the RL signal corrupts the embedding table).
- **Not for very short rollouts.** Diversity injection has less leverage when the trajectory is 1–2 tokens; benefits accrue with longer chain-of-thought.

---

## Sources

- Paper: *N-GRPO: Embedding-Level Neighbor Mixing for Enhanced Policy Optimization* — Zhu et al., Ant Group + Zhejiang U., 2026 — [arXiv:2606.10768](https://arxiv.org/abs/2606.10768).
