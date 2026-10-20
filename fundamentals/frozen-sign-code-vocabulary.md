# Frozen Sign-Code Vocabulary
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For small LMs, most parameters and most on-disk size live in the vocabulary table. The paper replaces that trained table with a **fixed 16 MB table of 512-bit sign codes**, derived once by projecting a pretrained embedding onto 512 principal components and taking signs. The LM body is trained from scratch against this frozen table; new words can be added **post-training** by projecting them through the same PCA and taking signs, with little degradation.

**Prereqs:** [_tokenization.md](_tokenization.md), [bpe.md](bpe.md)
**Related:** _(ties into the ternary / small-LM story in `quantization/_number-formats.md`)_

---

## What it is

For a 270M-class LM with a 256k vocabulary, the embedding table alone is roughly $256{,}000 \times 384 \approx 100\text{M}$ parameters — **most of the model**. Post-compression (e.g., 8-bit GGUF), the embedding still dominates file size. The paper's thesis: the embedding table doesn't have to be *learned* — it has to be a reasonable **sign code** that distinguishes tokens.

Derive the table once from a pretrained reference embedding, binarize it, freeze it, and train everything else from scratch against it. The embedding becomes a lookup into a tiny bit-packed table, not a learned parameter block.

## How it works

**One-time offline derivation.**
1. Take a pretrained embedding $E \in \mathbb{R}^{V \times d}$ for the same vocabulary (any public SLM works).
2. Fit **PCA** on $E$'s rows and keep the top 512 principal components.
3. Project every row onto those components and take **sign-binarization**: each row becomes a 512-bit code (one bit per component).
4. Pack the codes into a $V \times 512$-bit table. For a 256k vocabulary, that is $256{,}000 \times 512 / 8 \approx 16\text{ MB}$.

**Model body.** Train the transformer body from scratch with this table frozen. The embedding lookup produces a $\{-1, +1\}^{512}$ vector; a small learned projection maps it to the model's working width.

**Post-hoc vocabulary extension.** For a new word, embed it with the same pretrained reference model, project onto the stored PCA components, take signs. Add the resulting row to the table. No retraining of the body is required, and the new row is well-separated from existing ones because the sign code preserves large-component signs.

**Pair with ternary body.** The paper's full stack combines the frozen table with a **BitNet-style ternary** body for the attention and MLP weights.

## Why it matters

- **Shrinks the whole model.** With the frozen 16 MB table and a ternary body, the full model is a **46 MB file** that runs **5–6× faster** than the float version's 8-bit GGUF on a laptop CPU, with about **1/7 the RAM**.
- **Validation loss gap is small.** 0.20 nats higher than the learned-table baseline, with **2.6× fewer trainable parameters**. Matches or beats the learned baseline on LAMBADA, HellaSwag, PIQA; loses on ARC-Easy.
- **Rare-word handling improves.** Frozen tables are more robust on rare tokens because the sign code is derived from a bigger pretrained reference, not from the small training mix.
- **Post-hoc vocabulary extension.** Add **10,000 new words** with little degradation — a standing pain point for small models usually solved by continual pretraining.

## Gotchas & tricks

- **PCA source matters.** Derive from a strong, large pretrained embedding; derivations from under-trained or heavily-domain-specific embeddings transfer poorly.
- **Bit count is a knob.** 512 bits is the paper's recommendation; shorter codes (256) save more space but risk collisions on rare tokens.
- **Mild projection learnable.** A small learned linear map from the 512-bit code to model width lets the body adapt without the embedding being learned.
- **Not for frontier models.** The savings matter where the embedding is a large fraction of the model. For 70B-class models, the embedding is a rounding error.

## Sources

- Paper: *Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models* — Bathula, 2026 — [arXiv:2610.22428](https://arxiv.org/abs/2610.22428)
- Related: BitNet (ternary weights), PCA, sign-code binarization in information retrieval (LSH-style).
