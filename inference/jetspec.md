# JetSpec
*Depth — a causal-but-parallel speculative-decoding draft head that breaks the draft-budget scaling ceiling.*

**TL;DR:** Speculative decoding hits a soft ceiling as the draft budget grows: either acceptance drops (the draft tree contains inconsistent paths) or drafting cost balloons (depth-sequential autoregressive drafter). JetSpec resolves the dilemma with a **single-pass causal parallel draft head** trained on fused hidden states from the frozen target model — one forward pass produces a candidate tree whose joint distribution aligns with the target's autoregressive factorization. Up to **9.64× speedup on MATH-500** and **4.58× on chat** on Qwen3 dense and MoE.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md), [../pre-training/mtp.md](../pre-training/mtp.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md)

---

## What it is

A head-based speculative-decoding scheme that avoids the **causality-efficiency dilemma** prior heads face:

- **Autoregressive drafters** (e.g., EAGLE) produce path-conditioned candidates with high acceptance, but cost grows with tree depth — every additional draft layer is another forward.
- **Bidirectional block-diffusion drafters** generate all tree positions in one pass, but the resulting marginals are branch-agnostic. Each token looks plausible on its own, while the tree as a whole contains mutually inconsistent paths — wasted budget.

JetSpec is one-pass like a block-diffusion head and branch-causal like an autoregressive head.

## How it works

1. **Fused hidden states.** During target-model inference, extract intermediate hidden states from multiple layers and fuse them. These provide a richer conditioning signal than the final-layer hidden alone.
2. **Causal parallel draft head.** A trained head consumes the fused state and emits, in a single forward, the full draft tree — but with **branch-wise causal masking**. Each tree position is conditioned only on its ancestors (matching what an autoregressive drafter would see), so the joint distribution over the tree aligns with the target model's autoregressive factorization.
3. **Verification.** The target model verifies all tree positions in parallel as in standard SD. Because the draft tree's joint scores match the target's factorization, the longest accepted prefix is substantially longer than under bidirectional heads at the same budget.

The result: drafting cost is the bidirectional-head cost (one pass), acceptance length is the autoregressive-head behavior — the desired Pareto improvement on the budget-vs-speedup curve.

## Why it matters

- Breaks a soft ceiling that the SD literature has been hitting for two years.
- Particularly strong on **MoE** models, where expert routing makes per-token verification cheaper relative to re-decoding — the very regime where reasoning workloads live.
- Up to **9.64× on MATH-500** (long-CoT, the hardest workload to accelerate) and **4.58× on open-ended chat**, with further gains from vLLM integration under realistic serving loads.
- The "fused frozen hidden states + small trained head" recipe generalizes — similar in spirit to [MTP](../pre-training/mtp.md) heads — and is the right shape for the next generation of inference accelerators.

## Gotchas & tricks

- The head must be retrained per target model (and per fine-tune of that model). Costs are small but non-zero.
- Branch-wise causal masking is the load-bearing piece; ablating to flat causal masking collapses acceptance back to bidirectional-head levels.
- Gains compound with MoE-specific verification batching — non-MoE models see smaller (but still substantial) speedups.
- Code: [hao-ai-lab/JetSpec](https://github.com/hao-ai-lab/JetSpec).

## Sources

- Paper: *JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting* — Hu, Feng, Wu, Yuan, Zhao, Qian, Wang, Zhao, Jiang, Zhu, Rosing, Zhang, 2026 — [arXiv:2606.18394](https://arxiv.org/abs/2606.18394). UC San Diego / Zhejiang / UIUC / Nanjing / StepFun.
- Code: https://github.com/hao-ai-lab/JetSpec
