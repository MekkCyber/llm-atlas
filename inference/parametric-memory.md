# Parametric Memory (Deep Fusion Memory)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of retrieving relevant chunks and prefixing them to the prompt, train a small **causal predictor** to synthesize per-chunk "memory" from preceding context, with its target being the chunk's retrieved embedding. The frozen backbone LLM then ingests this memory through a gated cross-attention module layer by layer. At inference time there is **no retrieval step** — long-term memory lives entirely in hidden states.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

Retrieval-augmented generation (RAG) prefixes retrieved chunks to the prompt. Cost scales linearly with recall budget: more chunks = more context tokens on every forward pass. Parametric memory flips the design — build the memory *into* parameters/activations once, then use the backbone without any retrieval at inference.

Deep Fusion Memory (DFM) does this at **chunk granularity**: for every chunk the model will later see, a lightweight *memory predictor* produces an embedding-like memory token from the preceding context, trained to match that chunk's own retrieval embedding. These predicted memories are then injected into the frozen backbone through gated cross-attention.

---

## How it works

### Training

For each training chunk $c$ with preceding context $x_{<c}$:

1. Compute a target embedding $e_c = \mathrm{EncRet}(c)$ from a frozen retrieval encoder.
2. Run a small causal predictor over $x_{<c}$ to produce a memory vector $m_c = f_\theta(x_{<c})$.
3. Loss = distance between $m_c$ and $e_c$ (plus the backbone's next-token loss).

The predictor is small (orders of magnitude smaller than the backbone) and *causal* — it only reads context already available at that point in the sequence, so it can be reused unchanged at inference.

### Fusion into the backbone

The frozen backbone is augmented with **gated cross-attention** modules at selected layers. At each such layer:

```
h_t ──► self-attn ─► FFN ─► h'_t
          │
          └─► cross-attn( queries = h'_t, keys/values = m_c )
                       │
                       └─► gate · cross-attn-output + h'_t
```

The gate is a learned scalar (or small MLP) that lets the backbone decide, per layer and per position, how much of the predicted memory to pull in. The backbone itself stays frozen; only the cross-attention and gate are trained.

### Inference

At inference, there is no retriever and no prefix. The memory predictor runs online on the ongoing context to produce $m_c$ at chunk boundaries. The backbone consumes $m_c$ through the gated cross-attention. Context length used by the backbone stays unchanged.

---

## Why it matters

- **Constant inference cost.** Unlike RAG, which pays per retrieved chunk, parametric memory pays a fixed per-layer cross-attention cost that does not grow with how much corpus is "remembered."
- **No retriever at inference.** Removes retrieval latency and the retrieval-pipeline failure modes (bad chunker, bad ranker, stale index).
- **Keeps backbone frozen.** Can be bolted onto a shipped model without unfreezing weights — important for API-only deployments and for layering on top of expensive pretraining.
- **Reported gains across six QA and three domain-adaptation tasks** (code, math, law) over strong retrieval baselines.

---

## Gotchas & tricks

- **Not a free RAG replacement.** The predictor must be trained per corpus (or at least per domain). RAG is corpus-swappable at index-build time; parametric memory requires retraining when the corpus changes materially.
- **Retrieval-encoder choice is load-bearing.** The predictor is only as good as its target embedding. A weak retrieval encoder gives a weak target distribution; the predictor can't exceed it.
- **Capacity bottleneck.** A single memory vector per chunk compresses hard. For high-recall settings, multiple memory tokens per chunk (or hierarchical memory) are likely needed.
- **Composable with cache compression.** Parametric memory and KV-cache compression attack different costs (prefix tokens vs attention over long context) and should stack cleanly.

---

## Sources

- Paper: *Deep Fusion Memory: A Chunk-Level Parametric Memory for Large Language Models* — Wei, Sun, Sun, Liu, Shi, Wang, Zhang, Lin, SJTU LUMIA Lab / Shanghai AI Laboratory, 2026 — [arXiv:2610.25100](https://arxiv.org/abs/2610.25100).
- Precedent: *Memorizing Transformers* — Wu et al., 2022 — early non-parametric memory as kNN-attention.
- Precedent: *Retrieval-Augmented Generation* — Lewis et al., 2020 — the standard prefix-based retrieval baseline this contrasts with.
