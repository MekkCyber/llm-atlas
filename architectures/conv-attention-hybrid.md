# Convolution–Attention Hybrid Blocks
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A block-mix architecture that keeps full attention in only a minority of layers and uses **short convolutions with fixed-timestep memory** for the rest. The convolutional blocks never re-read a growing KV cache, so two-thirds of the network's per-token inference cost is constant in context length. **Daedalus-150M** (2026) uses 6 attention blocks and 12 conv blocks (2-timestep memory) across 18 layers; trained from scratch for CPU-only, 4-bit-weight inference, it beats similarly-sized transformers trained on 3–6× more data.

**Prereqs:** [transformer-block.md](./transformer-block.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [_moe.md](./_moe.md) · [../quantization/_number-formats.md](../quantization/_number-formats.md)

---

## What it is

The standard transformer stack pays attention cost on every layer, and its KV cache grows linearly with context. On a CPU decoding one token at a time, that cache re-read is expensive. A convolution–attention hybrid replaces most transformer blocks with a convolutional block whose "memory" is a **fixed, short window** (e.g. 2 timesteps) rather than a growing cache — cutting per-token cost drastically for long contexts.

## How it works

Daedalus's block mix (18 layers, 150M params):

- **Attention blocks (6/18).** Standard multi-head attention with a full KV cache. Placed at layers that most benefit from unrestricted context mixing (typically distributed through the stack, not clustered).
- **Convolutional blocks (12/18).** A short 1D depth-wise / gated conv over the token sequence. The relevant state at inference time is a **fixed 2-timestep tail** — no growing cache. Two-thirds of the network reads only the last 2 tokens of state per new token.

The receptive field of the whole network is still context-wide (attention layers propagate long-range info; conv layers refine it locally), but the *cache footprint* is dominated only by the attention layers.

Design decisions bundled with the block mix:
- Trained from scratch (no distilling from a big transformer) so that the model learns to use the block mix effectively.
- **4-bit weight storage** + **8-bit activations**, chosen at design time — the block mix is picked to still learn well under 4-bit weights.
- Optimizer/schedule tuned to the CPU-inference target from the start.

## Why it matters

- **CPU inference budget flips the design.** Instead of "make a good transformer, then squeeze it," Daedalus fixes the target (single-user, one token at a time, 4-bit, CPU) and picks the architecture around it.
- **Data efficiency at small scale.** Trained on 59.9B tokens, Daedalus-150M scores 47.31 on a fixed 5-task benchmark bar of 42.20 — beating GPT-2 124M, Pythia-160M, OPT-125M, GPT-neo-125M (each trained on 3–6× more data) and exceeding MobileLLM-125M (trained on ~1T tokens). Val bits-per-byte 0.8685.
- **Cache-free layers matter more as context grows.** The Daedalus argument scales: at 32K context, dense attention KV cost dominates; conv-hybrid layers stay flat.

## Gotchas & tricks

- **Ratio of attention to conv is the main knob.** More attention → more long-range capability but higher KV cost; more conv → cheaper cache but worse long-context reasoning. Daedalus at 6/18 is one operating point; the right ratio depends on target context length.
- **Distillation from a full transformer is a distinct approach.** Approaches like Zamba/Jamba distill or fine-tune from a pretrained transformer to warm-start the hybrid. Daedalus trains from scratch — cleaner comparison of the architecture itself, but requires the from-scratch budget.
- **Not the same as SSM hybrids (Mamba, Jamba, Zamba).** State-space hybrids replace attention with an SSM (structured recurrent operator) — different math from a short conv, but same broad "cheaper-than-attention block" philosophy.
- **Positional encodings must respect the block mix.** Conv blocks inject implicit position via kernel offset; attention blocks need RoPE / ALiBi / similar. Mixing without care produces inconsistent positional signals.
- **Quantization interacts with the block mix.** Some block types tolerate 4-bit weights better than others; picking the mix jointly with the target precision is part of the design.

## Sources

- Paper: *Daedalus-150M: A Convolution–Attention Hybrid Designed for CPU Inference* — 2026.
- Paper: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu & Dao, 2023 — SSM hybrid baseline.
- Paper: *Jamba* — AI21, 2024 — production transformer + Mamba hybrid.
- Paper: *MobileLLM* — Meta, 2024 — small-model design study for on-device inference.
