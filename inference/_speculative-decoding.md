# Speculative Decoding

*Taxonomy — accelerate autoregressive LLM decoding by drafting tokens and verifying them in parallel.*

**TL;DR:** A draft model proposes $\gamma$ candidate tokens; the target model verifies them in a single forward pass and accepts the longest valid prefix. End-to-end speedup is bounded by `τ · L_target / (T_draft + T_verify)`, so both **acceptance length** (draft quality) and **draft cost** matter. The design space breaks along two axes: **how the draft is produced** (autoregressive vs parallel vs head-based) and **how candidates are verified** (sequential vs tree-structured).

**Related taxonomies:** *(none yet — eventually `_kv-cache.md`, `_batching.md`)*
**Depth files covered here:** [domino-drafting](domino-drafting.md) · [draft-on-policy-distillation](draft-on-policy-distillation.md) · related: [../pre-training/mtp.md](../pre-training/mtp.md)

---

## The problem

Autoregressive decoding is memory-bound: each token requires loading all the model's weights, but the FLOPs done are tiny. GPUs are wildly underutilized. Speculative decoding fixes this by *amortizing* a single weight-load over multiple tokens — the target model's forward pass verifies several candidate tokens at once. The output distribution is provably preserved.

What goes wrong if you do this naively:
- **Cheap draft, low acceptance** → the speedup ceiling is small; verification overhead dominates.
- **Expensive draft** → drafting cost cancels the parallel-verify win.
- **Sequential draft loop** → $\gamma$ extra forward passes per cycle, kills throughput at long $\gamma$.

Every variant trades draft quality against draft cost differently.

---

## The shared pattern

```
prefix x_{≤t}  ─▶  draft model M_d  ─▶  γ candidate tokens
                                              │
                                              ▼
                target model M_t verifies all γ in one forward pass
                                              │
                                              ▼
                  accept longest valid prefix + 1 bonus token
```

Every variant has: a **drafter** that produces candidates, a **verification rule** (per-token rejection sampling that preserves the target distribution), and an **acceptance length** τ that determines speedup.

---

## Variants

| Technique | Drafting style | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Classic small-LM drafter** (Leviathan 2023, Chen 2023) | Smaller LM, autoregressive | Two models to maintain; small LM may diverge from target | Easy to deploy; works with any base model |
| **EAGLE / EAGLE-2 / EAGLE-3** (Li 2024–2025) | Single-layer head reuses target's last hidden state, autoregressive | Sequential $\gamma$ forwards through draft head | Highest acceptance length; standard for long-CoT |
| **Medusa** (Cai 2024) | Multiple parallel decoding heads on frozen base | Heads ignore each other → weak intra-block causality | Cheapest to bolt onto an existing model |
| **[MTP self-speculation](../pre-training/mtp.md)** (DeepSeek-V3 2024) | Extra prediction module trained jointly with pretraining | Requires pretraining-time investment | Built into the base model; no separate drafter |
| **DFlash / DART** (2026) — *no depth file* | Block-parallel non-AR drafter (diffusion-style) | Weak intra-block causal dependency | Removes the sequential-draft bottleneck |
| [**Domino**](domino-drafting.md) (2026) | Parallel backbone + low-rank causal-correction head | Slightly more params than pure parallel | Best balance of parallel + causal; new default |
| **FR-Spec / SpecVocab** (2025–2026) — *no depth file* | Static or dynamic vocabulary-subset projection | Coverage gap on rare tokens | Reduces LM-head projection cost on huge vocabs |
| [**Draft-OPD**](draft-on-policy-distillation.md) (2026) | Training-time fix: on-policy distillation for drafters | Requires verification-position replay infra | Trains any drafter beyond the SFT plateau |
| **SpecInfer / tree verification** (Miao 2023) — *no depth file* | Tree of candidates instead of single chain | Larger verification batch, more memory | Hedges over multiple plausible continuations |

---

## How to choose

**Default for new deployments (2026):** EAGLE-3 if you can tolerate the autoregressive draft loop and need maximum acceptance; Domino if you want the highest throughput at large concurrency (block-parallel + causal correction wins above 4-way concurrency).

**If you control pretraining:** bake in [MTP](../pre-training/mtp.md) — DeepSeek-V3-style depth-1 module gives you a calibrated, free drafter and slight downstream quality gain.

**If you can't retrain the drafter:** Medusa or a tiny donor LM. Cheapest to ship, lowest ceiling.

**Always pair with:** continuous batching, paged-attention KV cache, and tree-structured verification when memory permits. Speculative-decoding gains compose multiplicatively with batching gains.

**Training the drafter:** standard SFT on target-generated trajectories plateaus quickly because of the offline-to-inference mismatch — use [Draft-OPD](draft-on-policy-distillation.md) for on-policy distillation.

---

## Adjacent but distinct

- **[Multi-token prediction (MTP)](../pre-training/mtp.md)** — a pretraining objective; speculative decoding is one use. MTP modules *can* serve as drafters, but the technique is broader.
- **Parallel decoding / non-AR decoders** — generate without verification, sacrificing distributional correctness. Speculative decoding is provably distribution-preserving.
- **KV-cache compression** ([MLA](../architectures/mla.md), GQA) — reduces memory per token, not tokens per step. Orthogonal and composable.

---

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias, ICML 2023 — foundational draft-then-verify recipe.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., DeepMind, 2023.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE-3: Scaling up Inference Acceleration via Training-Time Test* — Li et al., NeurIPS 2025.
- Paper: *Domino: Decoupling Causal Modeling from Autoregressive Drafting* — Huang et al., 2026 — see [domino-drafting](domino-drafting.md).
- Paper: *Draft-OPD: On-Policy Distillation for Speculative Draft Models* — Lei et al., 2026 — see [draft-on-policy-distillation](draft-on-policy-distillation.md).
