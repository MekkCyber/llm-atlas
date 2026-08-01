# Speculative Decoding
*Taxonomy — draft-then-verify inference schemes that trade GPU cycles for wall-clock latency.*

**TL;DR:** Autoregressive decoding is memory-bandwidth-bound — the GPU spends most of every step reading weights and KV cache to produce one token. Speculative decoding runs a **cheap draft model** ahead in serial to propose $k$ tokens, then has the **target model verify** all $k$ tokens in a single parallel forward pass. Accepted prefix advances; rejected suffix is resampled. The family splits along two axes — *what produces the draft* (external small model, MTP head, retrieval, tree of drafts) and *how strict the verifier is* (lossless rejection sampling vs lossy relaxations). Lossless is the default; **lossy verification** trades small distributional drift for higher acceptance rates and needs careful design to avoid silent quality loss.

**Related taxonomies:** (none yet)
**Depth files covered here:** [lossy-verification](lossy-verification.md) · [mtp](../pre-training/mtp.md)

---

## The problem

At batch size 1 (or small batch), a modern LLM decode step is memory-bound: >90% of the wall-clock time is spent moving weights + KV cache from HBM to SRAM to produce a single token. The GPU's compute units sit idle. Any scheme that lets the target model **verify several proposed tokens in one forward pass** trades unused compute for latency, because the memory read cost of one forward pass is amortized across many tokens.

The catch: the accepted output distribution must match what plain decoding would have produced (**lossless**), or you must accept a controlled, characterized deviation (**lossy**). Naive relaxations degrade quality in ways that don't show up in aggregate benchmarks but do in the long tail.

## The shared pattern

```
1. Draft:   d_1..d_k  ← draft model (cheap, autoregressive)
2. Verify:  p_target(·|prefix), p_target(·|prefix, d_1), … in ONE parallel pass
3. Accept:  find the longest prefix d_1..d_j that passes the verifier's acceptance rule
4. Emit:    j accepted tokens + one resampled token from the target
5. Loop
```

Every variant is a different answer to two questions: **where does the draft come from?** and **when is a draft token accepted?**

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Vanilla speculative decoding** (Chen et al. 2023) — *no depth file yet* | Small external draft LM; **rejection-sampling** acceptance is exactly distributionally faithful. | Need to serve a second model. | The default lossless baseline. |
| [**MTP-based speculation**](../pre-training/mtp.md) | The main model's own multi-token prediction head produces the draft; no second model to serve. | Draft quality limited by MTP-head capacity. | Draft is free at inference (DeepSeek-V3 pattern). |
| **Medusa** (Cai et al. 2024) — *no depth file yet* | Multiple parallel "Medusa heads" on top of frozen target; tree-of-drafts verification. | Draft heads bolted on post-hoc; less signal per head than MTP. | Retrofitting speculation onto an existing model. |
| **EAGLE / EAGLE-2** (Li et al. 2024) — *no depth file yet* | Draft head predicts *hidden states*, not tokens; higher acceptance rate. | More complex draft training. | Highest acceptance rate in the lossless family. |
| **Lookahead decoding** (Fu et al. 2024) — *no depth file yet* | No draft model — Jacobi-iterate on n-grams instead. | Speedup smaller than model-based drafts. | When you can't ship / train a draft model. |
| [**Lossy verification (truncation-based)**](lossy-verification.md) | Accept draft samples inside a truncated target distribution. | Distributional distortion vs true truncated sampling. | Higher acceptance when nucleus/top-k is used anyway. |
| [**Lossy verification (collaborative)**](lossy-verification.md) | Draft + target combine acceptance rules; controlled overshoot. | Overshoot of draft probability degrades quality if unchecked. | When strict rejection sampling limits throughput. |

## How to choose

- **Default in 2026:** if the base model is trained fresh, ship it with an **MTP head** — the draft is free at inference and adds a training-signal bonus (DeepSeek-V3 pattern). If retrofitting, **EAGLE** currently has the best lossless throughput.
- **When latency matters more than exactness:** lossy verification can lift acceptance rates further, but only when you've explicitly characterized the failure mode. Read [lossy-verification](lossy-verification.md) before deploying — the failure modes are silent.
- **When you can't ship a draft:** lookahead decoding gives a modest speedup with zero training.

Combines with:

- **Continuous batching / paged attention** — orthogonal. Speculation reduces per-request latency; continuous batching improves throughput; use both.
- **Prefix caching** — orthogonal. Draft and target both benefit.
- **Prefill-decode disaggregation** — orthogonal, but speculation is mostly a decode-phase trick.

## Adjacent but distinct

- **Continuous batching** — improves throughput by co-scheduling many requests; doesn't reduce per-request wall-clock.
- **Non-autoregressive generation** — produces the whole sequence in one pass; sacrifices quality (very different tradeoff).
- **KV-cache compression / quantization** — attacks the *bandwidth* directly rather than amortizing it over multiple tokens.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., 2023 — the rejection-sampling formulation.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024.
- Paper: *Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes* — Wang et al., 2026 — [arXiv:2607.26627](https://arxiv.org/abs/2607.26627).
