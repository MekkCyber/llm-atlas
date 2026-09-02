# Off-Accelerator N-gram Embedding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Auxiliary embedding capacity added *outside* the accelerator's HBM: an **n-gram embedding table** — one embedding per short n-gram from a large vocabulary — is stored in **host memory** and prefetched per batch. Introduced in Qwen3.8-Next, which adds **51B parameters** of n-gram embeddings on top of 125B backbone parameters. The most-cited counter-intuitive finding: enlarging the n-gram vocabulary lowers pretraining loss *monotonically* while downstream accuracy *saturates* — a concrete case where loss-only ablations mislead.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md), [../fundamentals/bpe.md](../fundamentals/bpe.md)
**Related:** [../case-studies/qwen3-8-next.md](../case-studies/qwen3-8-next.md) · [qwen-sparse-attention.md](qwen-sparse-attention.md)

---

## What it is

Model embedding tables live on the accelerator (HBM). Their size is capped by HBM budget: bigger vocabulary means bigger table, which competes for HBM with attention KV cache and MoE expert weights.

But n-gram embeddings are **prefetch-friendly**: given a batch's input tokens you know exactly which n-gram entries will be needed, so they can be fetched from host memory to accelerator per batch (or per micro-batch) with predictable bandwidth. This unlocks table sizes that would be infeasible if the table lived in HBM.

Qwen3.8-Next adds **51B parameters** of n-gram embeddings this way — nearly half again the size of the 125B backbone — with modest overhead thanks to prefetching.

## How it works

**Table structure.** An n-gram vocabulary $V_n$ (n-grams of length ≤ some cap, filtered by frequency). Each n-gram $g \in V_n$ gets an embedding vector $e_g \in \mathbb{R}^d$. Table size is $|V_n| \times d$; scaled to 10s of billions of parameters, this doesn't fit HBM but comfortably fits host memory.

**Lookup at forward pass.** For each input token position, look up all n-grams ending at that position that appear in $V_n$. Sum (or gate) their embeddings into an auxiliary bias added to the standard token embedding.

$$
h_0 = E_{\text{tok}}(x_i) + \sum_{g \in V_n : g \text{ ends at } i} e_g
$$

**Off-accelerator storage + prefetch.** The full table sits in host memory. Per batch, the union of required n-gram IDs is computed on-device, sent to host, and the corresponding embeddings are fetched into a small HBM buffer. Because the input tokens are known before the forward pass, this fetch overlaps with compute.

**Backward pass.** Sparse updates on only the touched rows; gradients stream back to host memory for parameter updates.

## Why it matters

- **HBM-independent capacity.** Adding 51B parameters of embedding without spending 51B parameters' worth of HBM is a large win when HBM is the scarcity axis.
- **Extends the "add cheap capacity outside the backbone" pattern.** Similar in spirit to MoE (add expert capacity that's only partially activated per token), but for token-level n-gram statistics.
- **Latency-friendly.** N-gram lookup is deterministic given inputs; prefetch overlaps compute; per-batch HBM footprint is small.
- **Compatible with existing MoE + sparse-attention backbones.** Doesn't compete for HBM with attention or expert weights.

## Gotchas & tricks

- **Loss ≠ downstream.** Enlarging the n-gram vocab lowers pretraining loss monotonically because it captures token/short-context statistics the backbone would otherwise learn implicitly. Downstream accuracy saturates because those statistics don't help higher-level tasks past a point. **Do not scale the n-gram vocab by loss alone** — always verify with downstream benchmarks.
- **Host-memory bandwidth is the ceiling.** If your accelerator↔host link is thin, prefetch stops overlapping with compute and becomes the bottleneck. This works on H100-class + fast NVLink; less so on constrained interconnects.
- **N-gram filtering matters.** Naive n-gram enumeration explodes combinatorially; frequency-based filtering with a hard cap keeps $|V_n|$ tractable.
- **Interacts with tokenizer choice.** BPE tokenizers already collapse many common n-grams into single tokens, reducing the marginal value of n-gram embeddings. Larger vocab tokenizers help less.
- **Distributed training.** The full table replicated per host wastes memory; sharded across hosts requires an all-to-all-style fetch. Paper's specific arrangement not fully disclosed.
- **Not really an "embedding" in the semantic sense.** It's a learned bias table indexed by n-gram, not a compositional embedding of the n-gram's meaning. Treat it as a large parameter budget for token-context statistics.

## Sources

- Paper: *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability* — Qiu, Wang, Li, et al. — Qwen team / Alibaba, 2026 — arxiv.org/abs/2608.30320.
