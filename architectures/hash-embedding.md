# Hash-Signature Embeddings (MultiHashFormer)
*Depth — token representations as multi-hash signatures, decoupling embedding parameters from vocabulary size.*

**TL;DR:** Replace the standard `|V| × d` embedding matrix with a **hash signature** per token: a short sequence of discrete IDs produced by multiple independent hash functions. A small Hash Encoder packs the signature into a single latent for the Transformer, and a Hash Decoder predicts the next token's signature. Each token gets a *unique* signature (unlike one-hash methods), so causal LM training works, and parameter count no longer scales with `|V|`.

**Prereqs:** [_tokenization](../fundamentals/_tokenization.md), [bpe](../fundamentals/bpe.md)
**Related:** [transformer-block](../architectures/transformer-block.md), [attention](../fundamentals/attention.md)

---

## What it is

An embedding-side reparameterization that sits **between** the tokenizer (BPE, SentencePiece, …) and the Transformer. The tokenizer still produces an integer token ID; that ID is then mapped to a signature `(h_1(id), …, h_k(id))` using k independent hash functions. The Hash Encoder turns the signature into a vector; the Transformer is otherwise unchanged.

## How it works

**Forward.** For token id `t`:
1. Compute signature `s_t = (h_1(t), …, h_k(t))`, each `h_i(t) ∈ [0, B)` for a small bucket count `B`.
2. Look up `k` per-bucket embeddings and pass them through the Hash Encoder (a small Transformer/MLP) to produce a single latent `e_t ∈ R^d`.
3. Feed `[e_1, …, e_T]` to the standard Transformer decoder.

**Prediction.** The Hash Decoder reads the Transformer's hidden state and emits the *signature* of the next token (a short sequence of bucket IDs). The signature is then mapped back to the unique token via a precomputed inverse table.

**Why uniqueness works.** With `k` independent hashes and bucket count `B`, signatures collide with probability `1/B^k`; pick `B, k` so the entire vocab is collision-free. Earlier hash-based LMs only used `k=1`, which collapsed many tokens onto one vector and broke causal next-token training.

## Why it matters

- **Embedding/output cost flattens.** Standard LMs spend an outsized fraction of small-model parameters on input/output projections; hash signatures replace the `|V| × d` matrix with `k · B · d` (typically `≪ |V|`).
- **Multilingual expansion is free.** Adding a new language's vocab does not grow the parameter footprint — only the inverse table.
- **Empirical wins at 100M / 1B / 3B** scales against same-size standard Transformer LMs.

## Gotchas & tricks

- The Hash Encoder adds a small amount of FLOPs per token — usually negligible vs Transformer cost, but matters at very small scales.
- Inverse-table lookup at sampling time is `O(1)` if signatures are unique, but unique signatures require sizing `B^k` well above `|V|`.
- Behavior under aggressive tokenizer growth (e.g. 1M+ tokens for code+multilingual) is the main motivating regime; at small `|V|` the standard embedding still wins on simplicity.

## Sources

- Paper: *MultiHashFormer: Hash-based Generative Language Models* — Huiyin Xue, Atsuki Yamaguchi, Nikolaos Aletras — arXiv:2606.28057 — https://arxiv.org/abs/2606.28057
