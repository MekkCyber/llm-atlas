# SpecFold
*Depth — residual-gating-based branch folding for diffusion-LM speculative decoding.*

**TL;DR:** Diffusion language models verify multiple candidate branches in parallel during speculative decoding, and most of the per-branch compute duplicates its parent's work. **SpecFold** adds a learned **token-level residual gate** that reuses parent hidden states whenever a child branch barely changes them, and a hand-written **Triton kernel** that converts the saved work into wall-clock throughput. Reports up to **1.64× over Spiffy** and **1.99× over vanilla decoding**.

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [token-level-routing.md](token-level-routing.md)

---

## What it is

A speedup specifically for diffusion-language-model speculative decoding, where verification isn't a single draft but a tree of candidate branches. The tree is wasteful: nearby branches differ only in a handful of tokens, yet each gets a full forward pass. SpecFold's gating decides, per token, whether a child branch's hidden state can inherit from its parent (recomputation skipped) or must be fully recomputed.

Think of it as **computation folding** across a verification tree: identical sub-trees share work, near-identical sub-trees share work with a cheap residual correction.

## How it works

**Residual gate.** For each token position in a child branch, a lightweight gate $g_t \in \{0, 1\}$ (soft, with a straight-through trick during inference) is predicted from (a) the parent hidden state, (b) the differing input tokens, and (c) a learned threshold. If $g_t = 0$, reuse the parent's hidden state as-is; if $g_t = 1$, recompute.

Trained to predict when recomputation would barely move the hidden state, so the gate skips redundant work without changing the verified output distribution.

**Triton kernel.** The irregular reuse pattern — mix of fresh and inherited positions in the same batch — doesn't map onto PyTorch primitives. A custom Triton kernel does per-tile dispatch: positions with $g = 0$ copy from a parent buffer; positions with $g = 1$ run the full matmul. Memory access is kept regular.

The folding is applied layer-by-layer during the target model's parallel verification pass. The accept/reject rule for the final tokens is unchanged.

## Why it matters

Speculative decoding ported from autoregressive LMs to diffusion LMs underperforms because the verification tree, not a single draft, is the compute shape. SpecFold is the first speedup designed around that shape, and the folding idea generalises beyond diffusion LMs — any multi-branch rollout (best-of-N, tree-of-thought, search-based decoding) wastes work on near-duplicate paths.

Reported gains hold across two model families, five diffusion LMs, and five benchmarks at comparable task quality.

## Gotchas & tricks

- The gate must be trained *post-hoc* on real speculative-decoding traces, not generic text — the distribution of parent/child divergence is specific to the draft rule.
- Over-aggressive gating hurts quality subtly: the verified output distribution shifts even if accept rates stay stable. Monitor downstream task metrics, not just KL to full recomputation.
- Triton kernel assumes homogeneous model layer shape; non-uniform (MoE) layers need a specialised kernel per expert.
- Doesn't help when the verification tree collapses to a single branch (high draft accuracy) — SpecFold's gains scale with branch count.

## Sources

- Paper: *SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models* — 2026 — [arXiv:2610.04875](https://arxiv.org/abs/2610.04875).
- Related paper: *Spiffy* — the prior-SOTA diffusion-LM speculative decoder SpecFold compares against.
