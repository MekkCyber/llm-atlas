# Speculative Decoding

*Taxonomy — inference techniques that propose multiple draft tokens and have a strong model verify them in one pass, reducing the number of full-model decode steps.*

**TL;DR:** LLM decoding is memory-bandwidth-bound: each token costs one full forward pass even though most tokens are easy to predict. Speculative decoding splits the work into a cheap **draft** stage (multiple candidate tokens proposed by a small model, MLP heads, or the same model under a different policy) and a single **verify** stage where the strong model accepts a prefix in one parallel pass. When acceptance rates are high, multiple tokens are emitted per strong-model forward — a 2-4× wall-clock win without any quality loss when correctly implemented. The modern direction is **multi-tier** (different verification paths for different confidence regimes) and **draft-quality-aware** scheduling.

**Related taxonomies:** none yet — see [_sparse-attention](../architectures/_sparse-attention.md) for the orthogonal attention-cost axis.
**Depth files covered here:** [via-sd](via-sd.md)

---

## The problem

For autoregressive decoding, each output token requires a full forward pass through the strong model. The forward pass is dominated by **weight loading from HBM** (memory-bandwidth-bound), not by compute. The arithmetic intensity is terrible — most of the GPU's FLOPs go unused.

If you could verify multiple candidate tokens in one forward pass, the wasted FLOPs would do useful work. That's the opening speculative decoding exploits.

## The shared pattern

Every variant has the same skeleton:

1. **Draft**: propose $K$ candidate next tokens. Source varies: small model, self-drafting heads, prior context.
2. **Verify**: run one forward pass of the strong model in parallel over the drafted prefix. Accept the longest prefix where the strong model agrees with the draft (in a rejection-sampling-compatible way).
3. **Re-sample**: at the first disagreement, sample the strong model's distribution and continue.

Correctness is preserved because acceptance follows the rejection-sampling rule — the final token distribution is exactly what the strong model would produce decoding alone. Speedup comes from emitting *multiple* tokens per strong-model forward.

## Variants

| Technique | Draft source | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Classical (Leviathan / Chen) | Separate small model | Draft model maintenance + memory | Strong correlation between draft and target distributions |
| Medusa | Multiple MLP heads on the strong model | Heads need training; modest acceptance | No separate model to host |
| EAGLE | Auto-regressive draft head on strong model's hidden states | Higher acceptance, more complex training | Highest throughput on chat workloads |
| Lookahead | N-gram draft from context | No model at all | Highly repetitive workloads (code, structured output) |
| [via-sd](via-sd.md) | Multi-tier: confidence-routed slim verifier | Adds intra-model routing path | Mixed-confidence workloads, frontier models |

## How to choose

The modern default for serving is **EAGLE-style self-drafting** when you can afford the training and **Medusa-style heads** when you can't. Classical small-model drafting is fine when you already have a good small model in the family. **Lookahead** is a free win for code / structured generation but provides little for free-form chat. Multi-tier (VIA-SD) is the new direction when single-tier acceptance saturates — adding a slim verifier in the middle uses cheap compute on medium-confidence drafts that don't need the full model.

Speculative decoding stacks with **paged attention**, **continuous batching**, and **quantization**. It does *not* stack with itself: running two drafters in series is rarely worth it.

## Adjacent but distinct

- **Sparse attention** ([_sparse-attention](../architectures/_sparse-attention.md)) — reduces per-token attention cost. Speculative decoding reduces *number of tokens that pay the full cost*. Orthogonal.
- **KV cache compression** (MLA) — reduces memory per token. Orthogonal.
- **Parallel sampling / best-of-N** — runs multiple decoding streams in parallel for quality. Speculative decoding is about latency / throughput for a single stream.

## Sources

- Leviathan et al. (2023) — original speculative decoding formulation with the rejection-sampling correctness proof.
- Chen et al. (2023) — independent draft-verify formulation from DeepMind.
- Medusa (Cai et al., 2024) — MLP draft heads.
- EAGLE (Li et al., 2024) — auto-regressive draft head.
- VIA-SD — see depth file.
