# Output-Aware Block-Sparse Attention (ALLOT)
*Depth — allocate block-sparse attention budget by the queries' expected contribution to downstream outputs, not by their own self-similarity.*

**TL;DR:** Standard block-sparse attention selects which K/V blocks each query attends to using the query's own similarity map. **ALLOT** instead allocates sparsity by *output importance*: for each query block, estimate how much it will drive the tokens the user actually cares about, and give that block more or fewer keys accordingly. Training-free, swap-in for existing kernels, used in Video Diffusion Transformers where full 3D attention is prohibitive.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [stratified-sparse-attention](stratified-sparse-attention.md), [mla](../architectures/mla.md)

---

## What it is

Full 3D attention in Video Diffusion Transformers (video DiTs) is the compute bottleneck: attention over `(frames × height × width)` tokens scales quadratically with every dimension. Block-sparse attention is the standard workaround — divide tokens into blocks, and have each query block attend to only a chosen subset of key blocks.

Existing sparse-attention methods choose the subset by *input similarity*: for query block *q*, pick the key blocks most similar to *q*'s representation. This mirrors how dense attention would behave on average.

ALLOT argues this is the wrong objective. Not all query blocks matter equally for the output the user sees; some produce heavily-weighted output tokens (foreground motion), others produce background tokens whose final rendering is dominated by neighbors anyway. Allocating equal sparsity budget to all query blocks wastes budget on blocks whose output influence is small.

---

## How it works

For each query block *q*:

1. **Estimate output influence.** Compute a cheap proxy for how much *q*'s attention output will contribute to tokens the user perceives — in a video DiT this uses the magnitude of nearby-timestep gradients or the receptive-field footprint of *q*'s block.
2. **Allocate budget.** Give *q* a larger key-block budget (more keys to attend) if its output influence is high; smaller budget otherwise. Total budget is held constant across the layer.
3. **Pick keys inside budget.** Within the allocated budget, select keys by standard retrieval (top-k by similarity).

Key design choices:

- **Proxy must be cheap.** The proxy for output influence is computed from already-available signals (nearby denoising-step attention magnitudes), not a separate forward pass.
- **Budget normalization per layer.** Total keys per layer is conserved; ALLOT only redistributes.
- **No training.** The method plugs into existing block-sparse kernels; the sparsity *pattern* changes, nothing else.

---

## Why it matters

Video diffusion is the headline cost center for cheaper video generation at scale. Training-free sparsity that *doesn't degrade* quality is a direct speedup lever. Compared to input-similarity block-sparse baselines, ALLOT keeps quality at higher sparsity, which translates to end-to-end latency wins without re-training the model.

The idea is more general than video DiTs: in long-context LLMs, some query blocks also matter more than others for the output tokens the user reads (last block disproportionately so). Output-aware allocation likely transfers.

---

## Gotchas & tricks

- **Proxy quality is the ceiling.** A bad influence proxy allocates budget to the wrong blocks and underperforms input-similarity baselines. Validate the proxy on a held-out set before scaling.
- **Budget normalization matters.** Without a per-layer normalization, the method can accidentally raise *total* attention cost on some layers.
- **Not just for diffusion.** The paper's experiments are video DiTs, but the structural idea applies anywhere block-sparse attention is deployed.
- **Composability with retrieval tricks.** ALLOT selects *budget*; within each block, the key selection is still a retrieval problem — the two levers compose.

---

## Sources

- Paper: *Sparsity Should Be ALLOTted by Output: Training-Free Allocation of Block-Sparse Attention for Video Diffusion Transformers* — Tianze Xia, Xiaoling Zhou, Ruya Lan, Yiqiao Liao, Chengru Song, Shiming Xiang — PKU / KlingAI / UCAS, 2026 — [arXiv:2610.29441](https://arxiv.org/abs/2610.29441).
- Related: Native Sparse Attention (NSA); SpargeAttn; MInference and other long-context sparse-attention baselines.
