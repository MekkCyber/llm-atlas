# Speculative Decoding

*Taxonomy — accelerate autoregressive decoding by drafting candidate tokens with a cheap model and verifying them with the target in parallel.*

**TL;DR:** Autoregressive decoding is latency-bound by sequential token generation. Speculative decoding (SD) breaks the per-token barrier: a small/fast *drafter* proposes K tokens, the target model verifies them in a single parallel forward pass, and accepted tokens advance the cursor at no extra latency. Variants differ in *who drafts* (separate model vs head on the target) and *what shape they draft* (linear chain vs tree).

**Related taxonomies:** [_kv-cache-compression.md](_kv-cache-compression.md)
**Depth files covered here:** [jetspec.md](jetspec.md)

---

## The problem

Decoder transformers run one token per forward pass; latency scales with sequence length. On modern hardware the bottleneck is *memory bandwidth* (KV cache reads + parameter reads), not compute — a single decoder forward leaves most FLOPs on the table. Parallel verification of multiple candidate tokens recovers those wasted FLOPs *if* the candidates are mostly correct.

## The shared pattern

Every SD scheme has the same three pieces:

1. **Draft.** Produce K candidate tokens conditioned on the current context. Drafter must be much cheaper than the target.
2. **Verify.** Run the target on the K candidates in *one* parallel forward pass; obtain target logits for each position.
3. **Accept/reject.** Walk the candidates: accept while the target distribution agrees (under the SD acceptance rule); when it disagrees, accept up to that point, sample the corrected token from the residual distribution, and resume.

Speedup ≈ `expected accepted tokens per verification step / drafter cost ratio`. Two levers move it: *drafter quality* (acceptance length) and *drafter cost* (overhead per pass).

## Variants

| Technique | Drafter | Draft shape | Key idea | When it wins |
| --- | --- | --- | --- | --- |
| Vanilla SD (Leviathan / Chen 2023) | Separate small LM | Linear chain | Cheap external drafter; accept-reject ratio guarantees same output distribution | When a 100×-smaller compatible drafter exists |
| Medusa | Extra heads on target | Linear chain | Multiple "next-K" heads share target backbone; train heads on frozen target | Single-model deployment, simple integration |
| EAGLE / EAGLE-2 | Head conditioned on target hidden states | Tree | Reuse target's hidden states for the drafter; tree expands beams cheaply | Tight integration, high acceptance via tree search |
| Block-diffusion drafters | Bidirectional one-shot head | Tree | Emit all positions in one parallel pass | Cheap drafting; loses on long trees (branch-incoherent) |
| [jetspec](jetspec.md) | Causal parallel head over fused hidden states | Causal tree, one pass | One-pass tree that scores under target's autoregressive factorization | Large draft budgets that other methods can't convert to speedup |
| Lookahead decoding | None (uses past n-grams) | Linear chain | Recycles previous outputs as draft tokens via Jacobi iteration | Training-free, works with any model |

## How to choose

- **Default for new deployments: a head-based SD scheme on the target model.** Avoids the overhead of a second model and integrates cleanly with serving systems (vLLM, SGLang).
- Pick **tree** drafters (EAGLE, JetSpec) when you have spare verifier capacity per step and want the bigger draft budget to pay off; pick **chain** drafters (Medusa, vanilla) when verifier throughput is the bottleneck.
- Bidirectional block-diffusion drafters saturate on long trees because sibling tokens are scored independently — switch to causal tree drafters ([JetSpec](jetspec.md)) once tree depth exceeds a couple of layers.
- Lookahead is the cheapest to deploy (no training); use it as a baseline before reaching for trained heads.
- All schemes preserve the target's output distribution (same sampling, same temperature) as long as the accept-reject rule is followed correctly — speedup vs quality is *not* a tradeoff, modulo minor numerical drift.

## Adjacent but distinct

- **Multi-token prediction (MTP)** — see [mtp.md](../pre-training/mtp.md). Training-time objective that predicts multiple future tokens; pairs naturally with SD heads but isn't itself an inference-time acceleration.
- **Continuous batching** — packs multiple requests into one batch; orthogonal to SD and stacks with it.
- **KV-cache compression** — see [_kv-cache-compression.md](_kv-cache-compression.md). Reduces memory footprint, not per-token latency.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias, Google, 2022 — foundational SD with acceptance rule.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen, Borgeaud, Irving et al., DeepMind, 2023 — concurrent foundational work.
- Paper: *Medusa* — Cai et al., 2024 — head-based draft.
- Paper: *EAGLE / EAGLE-2* — Li et al., 2024 — feature-level draft + tree.
- Paper: *Lookahead Decoding* — Fu et al., 2024 — training-free Jacobi-based SD.
- Paper: *JetSpec* — Hu et al., 2026 — causal parallel tree drafting that breaks the scaling ceiling.
