# Loom Attention
*Depth — attention over head outputs via a per-token matrix memory (a second residual stream).*

**TL;DR:** Standard Transformers wire attention heads through a single residual stream — each head reads tokens, writes a delta back to that stream, and the wiring between heads is fixed by the architecture. Loom attention adds a *second* residual stream (a per-token matrix memory) that heads can also attend into, and over which they attend to each other's *outputs*. The resulting dual-stream Transformer (LoomFormer) matches standard Transformers trained with over 3× the compute, with ~2% extra FLOPs over the single-stream baseline.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [transformer-block.md](transformer-block.md)
**Related:** [attention.md](../fundamentals/attention.md) · [mla.md](mla.md)

---

## What it is

Mechanistic-interpretability work has shown Transformer computation is organized around *circuits* — groups of heads that compose predictably. Standard architectures fix which heads can compose: a head in layer L can only read what earlier heads wrote into the single residual stream. Loom attention lets the model learn to rewire these circuits per token by exposing head outputs directly to other heads through a second residual stream (a per-token matrix). "Loom" is apt: each token carries a small loom of threads that heads weave.

## How it works

For each token at each layer, the block maintains **two streams**:

- `x_t` — the usual per-token vector (the token residual stream).
- `M_t` — a per-token matrix (the head-output residual stream), dimensions roughly `d_head × k` where `k` is the number of threads.

Each head in the block does two attentions:

1. **Standard attention** over tokens using `x` as input, producing a value vector.
2. **Loom attention** over `M_t` (the token's own matrix memory), reading from the outputs of earlier heads in the same block or layer.

Both outputs combine additively into the head's write. The matrix `M_t` is updated at each block with the head's output as a new column; old columns age out or merge. Gradient flow through `M` gives the model a learnable wiring: a head can decide to read predominantly from the token stream, predominantly from its own prior heads, or from a mix, per token.

## Why it matters

- **Massive parameter efficiency.** LoomPythia-2.8B trained on 300B tokens matches or exceeds Pythia-12B on downstream tasks. Across scales, LoomFormer matches standard baselines trained with >3× compute.
- **~2% FLOPs overhead.** The matrix memory is small per token; the extra attention is cheap. Not a structural tax.
- **Mechanistic story is concrete.** The second stream is literally the medium through which heads talk to each other. This matches interp evidence that circuits (groups of heads) carry structured information — now the architecture reflects that.
- **Composable with existing attention variants.** Nothing stops Loom attention from using MLA-style KV compression or GQA on the token stream; the matrix stream is orthogonal.

## Gotchas & tricks

- **The matrix memory is per-token, not per-sequence.** You're not caching an interaction matrix across tokens — `M_t` is a small local memory. Don't confuse it with a global tensor.
- **Kernel support is custom.** No existing FlashAttention variant handles the dual-stream read. The paper ships a reference kernel; expect a few weeks of engineering to port to a new serving stack.
- **Loss landscape changes early.** Loom attention widens the family of representations the model can learn; early-training loss curves differ from a single-stream baseline. Don't tune LR/warmup from the baseline — retune.
- **Ablation: both streams matter.** Removing the standard attention (loom-only) collapses; removing the loom attention (standard-only) recovers the baseline. The gain is from the pairing.
- **Scaling claims are from one-paper runs.** The 3× compute match is impressive but is based on the authors' sweeps. Reproduction at >10B parameters is an open question.

## Sources

- Paper: *Rewirable Circuits in Transformers with Loom Attention* — Xiao, Li, Yuan (BUPT; ColorfulClouds), 2026 — [arXiv:2611.30634](https://arxiv.org/abs/2611.30634).
