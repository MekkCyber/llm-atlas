# Speculative Decoding

*Taxonomy — accept-or-reject sampling schemes that let a cheap "draft" model propose multiple tokens that a target model verifies in one forward pass.*

**TL;DR:** Autoregressive decoding is memory-bound on modern GPUs because one forward pass per token wastes compute on the KV read. Speculative decoding runs a *draft* (smaller/faster) model to propose $k$ tokens, then verifies all $k$ in a single target-model forward pass; accepted prefix plus one bonus token advance the stream. Modern variants differ in how the draft is parameterised (separate model, early-exit heads, a thin block stacked on the target's hidden states), how acceptance is scheduled, and how rejected drafts are reused. Lossless w.r.t. the target distribution at temperature 0.

**Related taxonomies:** [../architectures/_sparse-attention.md](../architectures/_sparse-attention.md) — orthogonal; both target decode latency.
**Depth files covered here:** [vspark](vspark.md) · [narrowspec](narrowspec.md)

---

## The problem

At batch 1, decode throughput is bounded by the speed of streaming the target model's parameters (and KV cache) through memory, not by compute. A single forward pass processes one token; the matmuls could do many more in the same wall-clock if more tokens were available. Draft-and-verify fills that compute with speculative tokens for free, provided acceptance is high.

## The shared pattern

```
Draft:     d(x_{<t})  →  propose  x̂_t, x̂_{t+1}, …, x̂_{t+k-1}
Verify:    T(x_{<t}, x̂_{t..t+k-1})  →  one forward pass over k+1 positions
Accept:    longest prefix where x̂_j matches a sample from T's distribution
Bonus:     one additional target-sampled token appended free of charge
```

Every variant has: (i) a draft, (ii) a draft-target alignment (how the draft's distribution maps to the target's), and (iii) an acceptance rule that keeps the output distribution exactly equal to the target's at matched temperature.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Vanilla SpecDec** (Leviathan 2023, Chen 2023) — *no depth file yet* | Separate small draft model, speculative sampling acceptance rule | Need to train / find a compatible draft | Baseline; works if a good smaller sibling exists |
| **Medusa** (Cai 2024) — *no depth file yet* | Multiple decoding heads on the target, each predicts the $k$-th future token | Heads must be trained; tree verification overhead | Deployed with the target as a single asset |
| **EAGLE / EAGLE-3** (Li 2024) — *no depth file yet* | One extra decoder block over the target's hidden states; feature-level draft | Heavier draft than Medusa; higher acceptance | Default for high-quality drafts when you can afford them |
| **DFlash** — *no depth file yet* | Output adapter re-uses the target's `lm_head` for exact logit equivalence | Only as good as its hidden-state draft | Settings that demand bit-level logit parity with the target |
| [**NarrowSpec**](narrowspec.md) (NVIDIA, 2026) | Draft residual width $\frac{1}{2}$ or $\frac{1}{4}$ of target width; fold output adapter into target's lm-head | Only tested with hidden-state drafts (DFlash, EAGLE-3) | Memory-tight hosts (edge, DGX Spark); up to 1.44× end-to-end |
| [**VSpark**](vspark.md) (KAIST, 2026) | Specialises speculative decoding for autoregressive VLAs: action-noise rollouts, hidden-feature alignment, rejected-draft reuse | VLA-specific training recipe | Robot action decoding; 1.28–2.10× lossless on 4 VLAs × 4 benchmarks |
| **Lookahead decoding** (Fu 2024) — *no depth file yet* | Parallel $n$-gram generation + Jacobi iteration — no draft model | Lower acceptance than trained drafts | When no draft is available / portable |

## How to choose

- **Have a good smaller sibling model?** Vanilla SpecDec or EAGLE-3 is the default — best acceptance, mature kernels.
- **Need one-asset deployment?** Medusa or EAGLE — the draft rides on the target's weights.
- **Memory-constrained host (consumer GPU, DGX Spark)?** Narrow the draft ([narrowspec](narrowspec.md)) — the draft is cheap enough that width barely matters for acceptance.
- **Non-text target (VLA, music, video tokens)?** Rebuild the draft training recipe for that modality; vanilla text-LM drafts under-accept. [VSpark](vspark.md) is the pattern for VLAs.
- **At high batch size?** Spec decoding gains shrink because compute-bound regime takes over; most gains come at batch 1–4. Verify empirically.

## Adjacent but distinct

- **Continuous batching / paged attention** (vLLM, SGLang). Orthogonal — bigger batches, not more tokens per step. Compose with speculative decoding.
- **KV cache compression / quantization.** Reduces per-token memory, doesn't change the token-per-step budget.
- **Prefill / decode disaggregation.** Routing-level optimisation; speculative decoding sits inside the decode stage.
- **Early-exit decoding.** Changes the target itself; speculative decoding leaves it intact.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., 2023.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE / EAGLE-3* — Li et al., 2024.
- Paper: *NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding* — NVIDIA, 2026.
- Paper: *VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models* — Yoo et al., KAIST AI, 2026.
