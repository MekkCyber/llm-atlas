# SigLIP

*Depth — CLIP's softmax contrastive loss replaced with per-pair sigmoid; enables larger batches with less memory.*

**TL;DR:** CLIP's softmax-InfoNCE requires an N×N similarity matrix with cross-batch normalization — memory and compute scale quadratically with batch size. **SigLIP** replaces it with a **per-pair sigmoid (binary cross-entropy) loss**: each (image i, text j) pair independently predicts "matched" or "not-matched." No cross-batch normalization → **memory linear in batch size** via chunked distributed implementation. Zhai et al. (ICCV 2023). Results saturate at batch ~32K (vs CLIP's 32K being the base). The default vision encoder for many 2024+ open VLMs (PaliGemma uses SigLIP SO400M).

**Prereqs:** [clip](clip.md), [vit](vit.md)
**Related:** [metaclip](metaclip.md) · [llava](llava.md)

---

## What it is

Zhai et al., *Sigmoid Loss for Language-Image Pre-Training*, ICCV 2023, arXiv 2303.15343.

Same architecture as CLIP (dual encoder, image + text), same data regime, **different loss**.

The observation: CLIP's softmax over the N×N similarity matrix forces you to compute and hold the full matrix, which grows as O(N²). At batch 32,768 this is already 1B entries. SigLIP's sigmoid loss is **per-pair** — each of the N² entries is independently binary-classified as "matched" or "not-matched." You can compute the loss chunk-by-chunk and sum, reducing peak memory.

---

## How it works

### The sigmoid loss (Sec. 3.2)

For a batch of size |B|:
- `x_i` = normalized image embedding.
- `y_j` = normalized text embedding.
- `z_{ij}` = +1 if i = j (matched), −1 otherwise.
- Learnable temperature `t = exp(t')`, initialized `t' = log 10` → `t ≈ 10`.
- Learnable bias `b`, **initialized b = −10**.

The loss:

```
L = −(1/|B|) · Σ_i Σ_j log( 1 / (1 + exp( z_{ij} · (−t · x_i · y_j + b) )) )

equivalently:

L_{ij} = −log σ( z_{ij} · (t · x_i · y_j − b) )
L = (1/|B|) · Σ_{ij} L_{ij}
```

where σ is the logistic function.

### Pseudocode (Algorithm 1)

```python
t = exp(t_prime)
z_img = l2_normalize(img_emb)         # [B, d]
z_txt = l2_normalize(txt_emb)         # [B, d]
logits = z_img @ z_txt.T * t + b       # [B, B], scaled by t, shifted by b
labels = 2 * eye(B) - ones(B)          # +1 on diagonal, -1 off-diagonal
loss = -sum(log_sigmoid(labels * logits)) / B
```

### Why the `b = -10` initialization

With |B| = 32,768, there are ~32K negatives for every positive in the batch. If b = 0 initially:
- σ(t · x · y) ≈ 0.5 for random pairs.
- Every negative pair contributes `−log(1 − 0.5) = log 2` to the loss.
- Summed: 32K × log 2 per row → huge gradient on negatives pulling everything down uniformly.

With `b = -10`, σ(t · x · y − 10) ≈ σ(−10) ≈ 4.5e-5 for all pairs at init — i.e., the model starts predicting "not a match" for everything. Matches the prior (most pairs aren't matched). Gradient on negatives is small; gradient on the diagonal positives is strong (they want to become matches). Gradient dynamics are much cleaner.

### Chunked distributed implementation (Sec. 3.3)

The critical implementation trick: instead of all-gathering all embeddings and materializing the full |B|×|B| matrix, each GPU keeps only its per-device batch `b = |B|/D`. The loss is computed in chunks:

1. Compute in-device loss (the `b × b` diagonal block on each device's "home" pairs).
2. Pass text embeddings around the ring — after D-1 rotations, every device has seen every other device's text embeddings.
3. At each step, compute the `b × b` cross-device loss block.
4. Sum all `b × b` block contributions.

**Memory**: O(b²) per device instead of O(|B|²) (CLIP's all-gather-based implementation).
**Communication**: one text-embedding rotation per step (D−1 steps total per batch).

This lets SigLIP scale to very large batches (paper goes up to 1M) on less memory than CLIP.

### Batch-size ablation (Sec. 4.2, Figure 3)

- Sigmoid beats softmax most strongly at batch sizes **<16K**.
- Both saturate around **batch 32K**.
- Beyond 32K: gains are negligible (1M gave no meaningful improvement).

Practical conclusion: **~32K is the sweet spot**. This is where CLIP operates; SigLIP matches CLIP there while being more memory-efficient to get there.

### Results

From the paper's Table 1:
- **SigLIP B/16** (32 TPUv4, 5 days, batch 32K): **73.4% ImageNet zero-shot**.
- **SigLIP g/14** (2 days on 4 TPUv4 for LiT variant with frozen backbone): **84.5%**.
- Later: **SigLIP SO400M** (shape-optimized, ~400M params) — widely adopted as the vision encoder for PaliGemma, Gemini-nano-vision variants, many open VLMs. Not from the SigLIP paper itself; from Alabdulmohsin 2023.

---

## Why it matters

- **Memory-efficient training at scale.** Chunked implementation lets you train at batch 32K+ without all-gathering all embeddings — critical on memory-constrained fabrics.
- **Better or equal quality at CLIP's batch size.** Sigmoid vs softmax gives small gains at 32K, larger gains at smaller batch sizes. Makes contrastive vision training work at smaller scales.
- **The default open vision encoder post-2023.** SigLIP SO400M in particular has become the go-to for open VLMs. PaliGemma, some Gemini variants, and most open-recipe VLMs from 2024 use it.
- **Cleaner theoretical story.** The per-pair sigmoid is simpler to reason about than softmax-with-temperature; the bias initialization trick is a specific, transferable insight.

---

## Gotchas & tricks

- **Use b = -10 at init.** This is load-bearing. Without it, the model's initial loss is dominated by "most pairs look similar" noise. The paper emphasizes this.
- **t = exp(t') initialization.** `t' = log 10` → `t ≈ 10`. Small temperature (compared to CLIP's τ ≈ 0.07 → 1/τ = 14) — similar magnitude.
- **Batch size still matters, just less.** SigLIP needs a reasonable batch size (8K+) to have enough negatives per positive. At very small batches, it degrades — but less than CLIP does.
- **Chunking implementation is non-trivial.** Requires careful overlap of communication and compute. Reference implementations in jax (Google's original) and PyTorch (OpenCLIP).
- **SO400M is "shape-optimized" not pure scale.** Its 400M params are arranged as a specific depth/width ratio found via neural-architecture-search-like experiments. Not from the SigLIP paper.
- **Temperature and bias are both learnable.** Both move during training. Keeping them frozen at their init values works but is slightly worse.
- **Compatibility with CLIP pretrained weights.** You can warm-start SigLIP from CLIP weights; the loss difference causes mild calibration drift that stabilizes in a few thousand steps.
- **Better gradient dynamics at small batch.** The softmax's cross-batch competition can cause instability; sigmoid's per-pair independence avoids this. Useful for fine-tuning with small batches.
- **Language coverage similar to CLIP.** SigLIP is English-first; multilingual variants exist but are less common than mCLIP.

---

## Sources

- Paper: *Sigmoid Loss for Language-Image Pre-Training* — Zhai, Mustafa, Kolesnikov, Beyer, ICCV 2023, arXiv 2303.15343.
- Paper: *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design* — Alabdulmohsin et al., 2023 — SigLIP SO400M.
- Paper: *PaliGemma: A versatile 3B VLM for transfer* — Beyer et al., Google, 2024 — canonical use of SigLIP SO400M as VLM vision encoder.
