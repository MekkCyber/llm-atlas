# Speculative Decoding
*Taxonomy — accelerate autoregressive LLM decoding by drafting and verifying multiple tokens per step.*

**TL;DR:** Vanilla autoregressive decoding emits one token per target-model forward — a hard ceiling on throughput. Speculative decoding (SD) breaks it by having a cheap *drafter* propose $K$ candidate tokens (often as a tree) and a *verifier* (the target model) accept-or-reject them in one parallel forward. Speedup tracks **acceptance length × budget efficiency**. The design space has two main axes: who drafts (separate small model vs head off the target's hidden states) and how the draft is structured (linear vs tree). The 2026 frontier is **head-based, parallel-tree drafters that are causal across branches** — the recipe JetSpec ships.

**Related taxonomies:** [_kv-cache-compression](_kv-cache-compression.md)
**Depth files covered here:** [jetspec](jetspec.md)

---

## The problem

Autoregressive decoding has structural latency: target-model latency per step × tokens per response. Throwing more GPU at it only helps if you batch — but batching hurts time-to-first-token and head-of-line latency. For latency-sensitive workloads (chat, reasoning agents), the only way to win is to **emit more tokens per target-model forward**.

Speculative decoding does exactly that: spend a cheap forward on a draft, spend the expensive forward on parallel verification, and amortize the target cost across however many drafted tokens get accepted.

## The shared pattern

Every SD variant has the same shape:

```
1. Drafter produces K candidate tokens (linear sequence or tree).
2. Target model verifies all K in one forward (parallel over positions).
3. Accept the longest prefix where verifier and drafter agree on top-1.
4. The next token after the accepted prefix is sampled from the verifier.
5. Repeat.
```

The acceptance criterion (greedy top-1 match, rejection sampling, or speculative sampling) guarantees the output distribution **matches the target model's distribution** — SD is lossless by construction.

Effective speedup $\approx$ (mean accepted prefix length) / (1 + drafter_cost / verifier_cost). Maximize the numerator without exploding the denominator.

## Variants

| Technique | Drafter | Tree? | Tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| Vanilla SD (separate drafter) | A smaller LLM | Linear | Drafter quality drift, two models to serve | Generic speedup with available smaller sibling models |
| Medusa | Multi-head off target hidden | Linear | Acceptance plateaus quickly | Lightweight drop-in, no separate model |
| EAGLE / EAGLE-2 | Autoregressive head off target hidden | Tree, causal | Drafting cost grows with tree depth | High acceptance, modest budgets |
| Block-diffusion / bidirectional head | One-pass parallel head | Tree, branch-agnostic | Inconsistent trees, budget waste | Cheap drafting, modest acceptance |
| **[JetSpec](jetspec.md)** | One-pass causal parallel head off fused frozen hidden states | Tree, branch-causal | Requires per-target head retraining | Reasoning (long-CoT) and MoE inference at large budgets |
| Self-speculative | Target model with skipped layers as drafter | Linear or tree | Smaller speedups; no separate weights | When deployment forbids extra weights |

## How to choose

- **Default to a head-based drafter** (Medusa-style, EAGLE-style, or JetSpec). Lower deployment overhead than a separate small model and tighter coupling to the target's distribution.
- **For large draft budgets and long outputs** (reasoning, agent traces), pick JetSpec or EAGLE-2 — bidirectional heads waste budget at depth.
- **For MoE targets**, SD is particularly valuable because verification batching amortizes routing overhead. Pair with JetSpec for the strongest gains.
- **vLLM / SGLang integrate** the head-based families; production deployments should look there first.
- If you can't afford to train a head (e.g., closed weights), a separate small drafter from the same family is the fallback.

## Adjacent but distinct

- **Multi-token prediction ([MTP](../pre-training/mtp.md))** — heads attached at *training time* that learn to predict $n$ tokens ahead. Architecturally close to SD heads; trained jointly with the target, used for both training-signal and inference acceleration.
- **Parallel decoding** without verification — non-speculative; drops the losslessness guarantee.
- **[KV cache compression](_kv-cache-compression.md)** — orthogonal lever; combine SD with cache compression for compounding speedups.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — original SD with speculative sampling.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees* — Li et al., 2024.
- Paper: *JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting* — Hu et al., 2026 — [arXiv:2606.18394](https://arxiv.org/abs/2606.18394).
