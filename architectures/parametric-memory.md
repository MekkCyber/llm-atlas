# Parametric Memory
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard decoder-only LMs entangle long-term memory and reasoning in one parameter set — you can't scale memory independently. **Parametric memory** splits them: a small base model pairs with a **large, dedicated memory decoder** pretrained on a retrieval-augmented next-token objective. Empirically, allocating parameters to memory beats scaling the base model — Pythia-410M + 6.9B memory matches Pythia-12B with 39% fewer total params.

**Prereqs:** [../pre-training/README](../pre-training/README.md), [transformer-block](transformer-block.md)
**Related:** [../fundamentals/dca](../fundamentals/dca.md)

---

## What it is

A parametric memory module is a full-size Transformer whose only job is to *store and retrieve* — its predictions are conditioned on kNN-retrieved neighbors from a large corpus. At inference, the memory decoder produces a distribution over next tokens grounded in retrieved evidence; the base model produces its own distribution; the two are combined (typically an interpolation) at each step.

Distinct from RAG: retrieval feeds a *learned parametric* module rather than being spliced into the base model's context. Distinct from external memory (kNN-LM): the memory decoder is a *scaled, pretrained* Transformer, not a lookup rule.

## How it works

**Pretraining objective.** For each training example, retrieve $k$ nearest neighbors from a Faiss index over the pretraining corpus. Train the memory decoder to predict the next token conditioned on the retrieved neighbors' distributions (a distillation-like objective on kNN posteriors).

**Scaling the pipeline.** At 6.9B params over 300B tokens, a single-machine Faiss index blows up. Memory Decoder at Scale contributes:
- **Distributed Faiss indexing / retrieval** across shards.
- **Sparse, batch-wise loading** of kNN distributions — only load the top-k neighbor rows needed per batch, not the full posterior tensor.

**Inference.** Base model $p_{\text{base}}(x_t \mid x_{<t})$ and memory decoder $p_{\text{mem}}(x_t \mid x_{<t})$ produce distributions in parallel. Combine via learned interpolation or fixed mixing coefficient.

**Domain memories.** A single general-purpose memory can be swapped for a **domain-specific** memory (e.g. medical, code) trained on the domain corpus. Small (1.7B) domain memories yield >9-point avg gains across three domains at every base scale from Qwen3-0.6B to Qwen3-14B.

## Why it matters

- **Parameter-efficient scaling.** 6.9B memory + 410M base beats 12B monolithic on 17 benchmarks — 39% fewer params.
- **Independent memory scaling** — grow memory without retraining the base; add domain memories per deployment.
- **Reopens the "small reasoner + huge memory" architecture** as a serious frontier design, with implications for serving cost, on-device inference, and continual updates.

## Gotchas & tricks

- Faiss retrieval cost is the bottleneck at scale — the distributed sharded index is essential above ~1B memory size.
- Sparse batch loading is a memory-layout trick, not an algorithmic change; without it, the kNN posterior tensor doesn't fit in RAM.
- Interpolation coefficient between base and memory distributions is tunable per domain; a fixed value works surprisingly well.
- Memory decoders benefit from *pretraining* on the retrieval objective — bolting kNN onto a standard-pretrained decoder underperforms.
- Domain-specific memories can be much smaller than the general one and still add substantial value.

## Sources

- Paper: *Memory Decoder at Scale: A Pretrained, Parametric Long-Term Memory* — Wei et al., Shanghai AI Lab / SJTU, 2026 — [arXiv:2607.27919](https://arxiv.org/abs/2607.27919).
- Precursor: *kNN-LM: Generalization through Memorization* — Khandelwal et al., 2020.
