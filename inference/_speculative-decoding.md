# Speculative Decoding

*Taxonomy — draft-and-verify schemes that turn one target-model forward pass into many accepted output tokens.*

**TL;DR:** Autoregressive decoding is bandwidth-bound: every token reloads the full model. Speculative decoding (SD) breaks the 1-token-per-step ceiling by having a cheap drafter propose several candidate tokens (or a tree of candidates), then verifying all of them with **one** target-model forward pass. Accepted prefixes commit; rejected suffixes get re-rolled. The family splits along three axes — draft architecture (separate model vs. head-on-target), draft shape (linear vs. tree), and draft causality (autoregressive vs. bidirectional/parallel).

**Related taxonomies:** [_number-formats](../quantization/_number-formats.md)
**Depth files covered here:** [jetspec](jetspec.md)

---

## The problem

LLM decoding is memory-bandwidth-bound on modern accelerators: each step streams the entire weight matrix to compute one token. Almost all of that bandwidth is wasted relative to the FLOPs available. SD trades a small amount of *extra* compute (the draft + the per-step verification of K candidates instead of 1) for much higher *utilization* of the bandwidth already paid for.

The ceiling on SD speedup is set by **acceptance rate × drafter overhead**. Push acceptance higher and the drafter must be smart; make it smart and it gets slow. Every SD variant lives somewhere on that tradeoff.

---

## The shared pattern

```
loop until done:
  drafter → K candidate tokens (or a tree)
  target  → 1 forward pass scoring all K positions in parallel
  accept the longest prefix that matches the target distribution
  resample the first mismatched position from a corrected distribution
```

The acceptance criterion preserves the target model's *exact* sampling distribution — this is the key guarantee that makes SD strictly an inference optimization rather than a quality tradeoff.

---

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Vanilla SD (Leviathan / Chen) | Small autoregressive draft model | Two separate models to host; linear chain | High target-drafter alignment, long reasoning |
| Medusa | Multiple MLP heads on the target model | Per-position marginals; no branch coupling | Cheap deployment alongside target |
| EAGLE / EAGLE-2 | Autoregressive head fed by target hidden states | Path-conditioned draft tree; head still autoregressive | Strong acceptance, moderate draft cost |
| Block-diffusion drafters | One forward pass emits all draft positions | Mutually inconsistent trees; wasted budget | Maximum drafter throughput |
| [jetspec](jetspec.md) | Causal parallel tree drafting from fused target hidden states | Single forward pass *and* branch-causal scoring | Long-CoT reasoning; large draft budgets |
| Lookahead decoding | Use n-gram cache from current generation as drafter | No training; weaker on novel prompts | Serving frameworks with no extra weights |

---

## How to choose

- **Default for new deployments:** EAGLE-2 or JetSpec on top of the target model. Both avoid a second hosted model and both build trees rather than chains.
- **Long-CoT / reasoning workloads:** prefer head-on-target SD with tree drafting — acceptance is naturally high and draft budgets pay back. JetSpec is the current frontier on this regime.
- **Cheapest path:** lookahead decoding gives a free ~1.5–2× without training.
- **Two-model SD** (separate small drafter): still useful when the target is an opaque API or when the drafter can be trained on a very narrow distribution that lifts acceptance to near 1.

Rule of thumb: end-to-end speedup ≈ `accepted_length / (1 + drafter_cost / target_cost)`. JetSpec's contribution is moving both terms in the right direction at once.

---

## Adjacent but distinct

- **[Multi-token prediction (MTP)](../pre-training/mtp.md)** — a *training* objective that makes the base model emit several tokens jointly; SD reuses those heads at inference time as drafters in some recipes (e.g., DeepSeek-V3).
- **Continuous batching** — orthogonal throughput lever at the scheduler level; composes cleanly with SD.
- **Cache compression** (e.g., [kv-cache-compression.md](kv-cache-compression.md)) — orthogonal latency lever on the KV-side; composes with SD.

---

## Sources

- *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022 — the canonical formulation.
- *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., 2023 — concurrent primary source.
- *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees* — Li et al., 2024.
- *JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting* — Hu et al., 2026 — [arXiv:2606.18394](https://arxiv.org/abs/2606.18394).
