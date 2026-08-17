# Hybrid linear attention

*Depth — interleaving full-attention layers with cheap linear-attention (or SSM) layers to trade quality for cost on a per-layer basis.*

**TL;DR:** A **hybrid linear-attention (HLA)** LLM keeps most of its layers cheap (linear attention or a state-space block) and inserts a small number of **full-attention layers** at fixed depths. The full layers provide the "recall bandwidth" the linear layers lack; the linear layers provide the O($n$) inference cost. Every modern long-context frontier model in 2025–26 (Jamba, Zamba, RecurrentGemma, MiniMax-01, several 397B-scale open models) uses some variant of this design. The design choice that matters most is the *ratio* and *placement* of full layers.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [transformer-block](transformer-block.md)
**Related:** [mla](mla.md), [sliding-recurrent-memory](sliding-recurrent-memory.md), [../interpretability/massive-activations.md](../interpretability/massive-activations.md)

---

## What it is

Full self-attention is O($n^2$) in sequence length. Linear-attention variants (Performer, Linear Transformers, gated deltas like GDN, and state-space blocks like Mamba/S4) reduce this to O($n$) at the cost of associative-recall accuracy — they struggle to look up specific past tokens by content. **Hybrid** designs stop trying to pick one: they use linear layers as the default and drop in a small number of full-attention layers, spaced regularly through depth, to provide targeted recall.

The design axes are:

- **Density of full layers** — 1 in 6, 1 in 8, 1 in 16; the fewer full layers, the cheaper inference but the weaker recall.
- **Placement** — top-heavy, bottom-heavy, uniform, or "sandwich" (first and last).
- **Linear-block choice** — Mamba/S4, GDN (Gated DeltaNet), RWKV, RetNet, and their gated variants.
- **KV sharing** — some hybrids share KV cache across neighboring full layers to further reduce memory.

## How it works

A typical HLA stack:

```
[ linear ] × k1
[ full   ]
[ linear ] × k2
[ full   ]
...
[ linear ] × kn
```

At training time, backprop flows normally through both block types. At inference:

- **Full layers** maintain a standard KV cache — this is where memory grows linearly in seq length.
- **Linear layers** maintain a fixed-size recurrent state (matrix or vector), so their memory footprint is O(1) in seq length.

Total inference memory ≈ (# of full layers) × KV per full layer + O(1) × (# of linear layers). Getting the "# of full layers" fraction right is the whole optimization.

Empirically (Su et al., 2026) hybrid models sit on a continuum between full-attention and pure-linear models: as full-attention density → 1, hybrids recover the behavioral fingerprints of dense transformers, including **massive-activation morphology** (see [massive-activations](../interpretability/massive-activations.md)). As density → 0, hybrids inherit the recall failures of pure linear models.

## Why it matters

- **Long-context inference cost is the dominant serving concern for reasoning models.** HLA is the current architectural answer.
- **Better recall than pure linear, cheaper than full.** For a fixed serving budget, HLA is on the Pareto frontier of quality × cost for context lengths beyond ~32k.
- **Composable with everything downstream.** Quantization, speculative decoding, prefill/decode disaggregation all work on HLA the way they work on full-attention — just factor per-layer.
- **Emergent structure is architecture-aligned.** The pre-attention spike + inter-spike plateau morphology described in the 2026 study means quantization and outlier-handling recipes can be *positional*, not global — protect the layer feeding into each full layer, permit larger dynamic range through the linear stretches.

## Gotchas & tricks

- **Full-layer placement matters more than density.** Two hybrids with 1-in-8 full layers can differ significantly if one puts them uniformly and the other clusters them at the top.
- **Massive activations concentrate around the full layers.** Ignoring this during INT4/FP4 calibration leaves the biggest per-layer outliers on the table. See [massive-activations](../interpretability/massive-activations.md).
- **Long-context evals mislead.** Perplexity on long contexts is dominated by short-range prediction; needle-in-a-haystack and multi-key retrieval tests actually stress the full layers. Report both.
- **Ablation stability.** Removing a single full layer from a well-placed hybrid can catastrophically damage recall. Removing a linear layer often barely moves the loss. Recall capacity is not evenly distributed.
- **Training instability at the transition.** Some hybrids show loss spikes when full and linear blocks disagree strongly. QK-norm on the full layers is a common fix.

## Sources

- Paper: *Massive Activations in Hybrid Linear Attention Large Language Models* — Su, Sun, Zhuang, Zhang, Xiao, Xiong, Zhang, Zhou, Zhang, Wong, Kuo, 2026, [arXiv:2608.12149](https://arxiv.org/abs/2608.12149) — the systematic characterization of HLA behavioral morphologies.
- Paper: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu & Dao, 2023, arXiv 2312.00752 — a common linear block used in hybrids.
- Paper: *Jamba: A Hybrid Transformer-Mamba Language Model* — AI21 Labs, 2024 — one of the earliest production-scale hybrids.
