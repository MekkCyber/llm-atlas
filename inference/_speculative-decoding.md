# Speculative Decoding

*Taxonomy — serve one target model's output faster by letting a cheaper drafter propose tokens the target then verifies.*

**TL;DR:** Autoregressive decoding is memory-bound: each new token requires a full model forward pass to produce one output. Speculative decoding breaks that one-to-one ratio by having a **drafter** (a smaller model, extra model heads, or a cheap predictor) propose a short sequence of candidate tokens, and the target model verifies them **in parallel** in one forward pass. Accepted tokens cost roughly one target pass each; rejected ones fall back to standard decoding. The family splits along what the drafter is (separate small model, same-model head, speculative sampling from prior outputs) and what gets verified (next tokens for AR models, commit *order and content* for diffusion LMs).

**Related taxonomies:** *(none yet — the `inference/` folder is new)*
**Depth files covered here:** [spot-speculative-decoding](spot-speculative-decoding.md)

---

## The problem

AR decoding is bottlenecked by per-token memory bandwidth: loading model weights from HBM dominates the latency of producing one token. The compute arithmetic intensity at batch 1 is terrible — the model is doing a huge matmul to emit a single vector of logits. If we could reuse one forward pass to emit **several** tokens, we'd amortize that bandwidth over multiple outputs.

Diffusion language models have a different variant of the same problem: each diffusion step is a full forward pass, and committing one token per step wastes most of the pass's output. Any speedup must handle the extra choice of *order* on top of content.

## The shared pattern

```
drafter proposes k candidate tokens  ──▶  target model scores all k in parallel
                                                     │
                                                     ▼
                                        accept the longest matching prefix,
                                        resample at the first disagreement
```

Every speculative-decoding variant answers four questions:

1. **Who is the drafter?** A separate small model, a side-head of the target, the target's own earlier predictions, or a learned pattern.
2. **How many candidates per round?** Fixed $k$ or dynamic.
3. **How is acceptance decided?** Exact match on token, or probabilistic rejection sampling that preserves the target distribution.
4. **Does the drafter also commit *where*?** For AR, position is fixed by causality. For diffusion LMs, the drafter must predict both *what* and *where* ([SPOT](spot-speculative-decoding.md)).

## Variants

| Technique | Drafter | What's predicted | Preserves target distribution? | When it wins |
| --- | --- | --- | --- | --- |
| **Classical speculative decoding** (Chen 2023, Leviathan 2023) — *no depth file yet* | Separate small AR model | Next tokens | Yes (rejection sampling) | AR LMs where a cheap-but-correlated drafter exists |
| **EAGLE / EAGLE-2** (Li 2024) — *no depth file yet* | Side head on target's hidden state | Next tokens (autoregressive on features) | Yes | Highest accept rate among AR drafters; needs training |
| **Medusa** (Cai 2024) — *no depth file yet* | Multiple parallel heads on target | Next few tokens, tree verification | Approximate (greedy); typical-acceptance mode preserves dist. | Simplest to train; strong for greedy decoding |
| **Multi-token prediction** ([../pre-training/mtp.md](../pre-training/mtp.md)) | Future-token heads trained during pretraining | Future tokens as side task | n/a (training time) | Builds-in drafter capability from pretraining |
| **Self-speculative decoding** — *no depth file yet* | Early layers of the target itself | Next tokens | Yes | No separate drafter weight; higher accept rate ceiling than tiny model |
| **[SPOT](spot-speculative-decoding.md)** (Shlomi 2026) | 0.62M-param drafter on target's distribution | Commit **order and content** for diffusion LMs | Yes (within the DLM's commit mechanics) | **Diffusion** LMs — the only family with order-prediction |

## How to choose

**For AR LMs with available smaller models in the same family** (e.g. Llama-7B drafting for Llama-70B), classical speculative decoding is the baseline. Picking the drafter is more art than science; a drafter trained on the same data as the target wins over one trained independently.

**For AR LMs without a natural drafter**, **EAGLE** is the current open default — highest token-accept rate, side-head training is cheap. **Medusa** is simpler if you only care about greedy decoding or can tolerate approximate distribution.

**At training time, if you control the recipe**, [multi-token prediction](../pre-training/mtp.md) builds drafter capacity into the model itself. Avoids any inference-time drafter.

**For diffusion LMs**, [SPOT](spot-speculative-decoding.md) is the only published approach that handles the order-prediction axis. Standard AR speculative decoding doesn't apply — the model emits content *and* positions each step.

**Across all variants**, the key metric is **average accepted length per target pass**. Measure it on your workload; the "typical" accept rate reported in papers often doesn't match production prompts.

## Adjacent but distinct

- **Multi-token prediction heads at training time** vs. speculative decoding at inference time. MTP is a *source* of fast drafters; speculative decoding is the inference mechanism that uses them. See [../pre-training/mtp.md](../pre-training/mtp.md).
- **Draft-and-verify over a large batch** is just regular batching, not speculative decoding. SD specifically trades batch for *sequence-length* parallelism in one request.
- **Continuous batching / inflight batching** (vLLM, SGLang) is orthogonal; it maximizes throughput by overlapping requests but doesn't change per-request latency. SD does.
- **KV-cache compression and paged attention** reduce memory footprint; SD reduces number of target-forward-passes per output token. Both compose.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — the canonical speculative-decoding paper for AR LMs.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., 2023 — concurrent formulation.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024 — side-head drafter on hidden states.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction* — Shlomi, Rahimi, Mendelson, Baskin, 2026 — [arXiv:2610.22543](https://arxiv.org/abs/2610.22543) — the DLM extension.

---

## Conventions

- **Filename:** `_speculative-decoding.md` (leading underscore — taxonomy).
- **Folder placement:** `inference/` root, alongside its depth files.
- **Scope:** inference-time techniques that break the "one target pass per token" ratio. Training-time techniques that enable drafters (MTP) are related but belong in their training folder.
