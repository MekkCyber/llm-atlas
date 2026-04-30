# HuBERT

*Depth — self-supervised speech pretraining via iterative k-means pseudo-labels and BERT-style masked prediction.*

**TL;DR:** Hsu et al. (2021). Like wav2vec 2.0 but with **offline k-means clustering** as the quantizer instead of learned-codebook + Gumbel-softmax. Iterate: (1) cluster features with k-means to produce pseudo-labels, (2) train HuBERT to predict these labels at masked positions, (3) use HuBERT's own intermediate-layer features to re-cluster for the next iteration. **Iteration 1**: k=100 k-means on MFCCs. **Iteration 2**: k=500 on layer-6 HuBERT features. **Iteration 3**: layer-9 features of iter-2 Base for Large/X-Large. X-Large (964M) hits 4.6 / 6.8 on LibriSpeech test-clean / test-other. More stable to train than wav2vec 2.0; more complex than BEST-RQ.

**Prereqs:** [mel-spectrogram](mel-spectrogram.md), [wav2vec2](wav2vec2.md)
**Related:** [best-rq](best-rq.md) · [conformer](conformer.md)

---

## What it is

Hsu, Bolte, Tsai, Lakhotia, Salakhutdinov, Mohamed, *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*, arXiv 2106.07447, 2021.

Motivated by the observation that wav2vec 2.0's **learned codebook** is brittle — collapse failures, Gumbel-softmax tuning, diversity loss, etc. HuBERT's alternative: **generate pseudo-labels via k-means**, then train via pure BERT-style masked prediction (cross-entropy on cluster IDs). No learned codebook, no contrastive loss, no Gumbel-softmax.

Key insight: iterate. The first k-means is on poor features (MFCCs); the resulting HuBERT's features are better, so re-cluster on those; the next HuBERT is better still. **Iterative refinement**.

---

## How it works

### Masked prediction loss (Eq. 1)

For masked position t with true cluster ID `z_t`:

```
L_m(f; X, M, Z) = Σ_{t ∈ M} log p_f(z_t | X̃, t)
```

where `X̃` is the masked input and `M` is the set of masked positions. `L_u` is the same summed over unmasked positions. Final loss:

```
L = α · L_m + (1 − α) · L_u
```

HuBERT uses **α = 1** — loss only on masked positions (BERT-style). Authors argue this is more resilient to noisy cluster targets than mixed-position loss.

### Predicted distribution (Eq. 3)

```
p_f(c | X̃, t) = exp(sim(A·o_t, e_c) / τ) / Σ_{c'} exp(sim(A·o_t, e_{c'}) / τ)
```

where:
- `o_t`: Transformer output at position t.
- `A`: projection matrix (to bring `o_t` into the codeword embedding space).
- `e_c`: codeword embedding for cluster c.
- `τ = 0.1`: temperature.

This is a standard softmax classifier with cosine-similarity logits. Unlike wav2vec 2.0's contrastive loss, HuBERT uses full softmax over all clusters.

### Iterative refinement (Sec. IV-B)

**Iteration 1**:
- Compute **39-dim MFCCs** (13 MFCCs + Δ + ΔΔ) from raw audio.
- Run **k-means with k = 100 clusters**.
- Train HuBERT Base for 250K steps, predicting cluster IDs.

**Iteration 2**:
- Extract **layer-6 output** features from iter-1 HuBERT Base on the training data.
- Run **k-means with k = 500 clusters** on those features.
- Train HuBERT Base for 400K steps on the new labels.

**Iteration 3** (for Large / X-Large):
- Extract **layer-9 output** of iter-2 Base on 60K hours of Libri-Light.
- Cluster.
- Train HuBERT Large (or X-Large) on these labels.

K-means: MiniBatchKMeans (scikit-learn), batch 10,000 frames, k-means++ init with 20 restarts.

### Architecture (Table I)

| Spec | Base | Large | X-Large |
|---|---|---|---|
| Transformer layers | 12 | 24 | 48 |
| Embedding dim | 768 | 1024 | 1280 |
| FFN dim | 3072 | 4096 | 5120 |
| Attention heads | 8 | 16 | 16 |
| LayerDrop | 0.05 | 0 | 0 |
| Projection dim | 256 | 768 | 1024 |
| Params | 95M | 317M | **964M** |

CNN feature encoder is **identical to wav2vec 2.0**: 7 layers, 512 channels, strides (5, 2, 2, 2, 2, 2, 2), kernel widths (10, 3, 3, 3, 3, 2, 2). 20 ms frame rate at 16 kHz (320× downsampling).

Transformer otherwise follows wav2vec 2.0 but with LayerDrop (in Base).

### Masking (Sec. IV-C)

Same as wav2vec 2.0: span mask with length l = 10, start probability p = 8%. SpanBERT-style masking.

### Training

- Adam (β = (0.9, 0.98)), linear warmup 8% of total steps then linear decay.
- Peak LR: 5e-4 / 1.5e-3 / 3e-3 for Base / Large / X-Large.

### Fine-tuning

- Projection head replaced with random softmax.
- **CTC loss** over 26 letters + space + apostrophe + CTC blank.
- CNN encoder **frozen**.
- Transformer layers unfrozen.

### Results (Sec. V)

- Base matches / slightly beats wav2vec 2.0 Base at the same scale on LibriSpeech.
- Large: 4.7 / 7.6 test-clean / test-other (no LM), 0.1 / 0.6 below wav2vec 2.0 SOTA.
- **X-Large (964M, 60K h Libri-Light)**: 19% / 13% relative WER reduction on dev-other / test-other vs Large. Example: **4.6 / 6.8 test-clean / test-other**.

---

## Why it matters

- **Cleaner than wav2vec 2.0.** No Gumbel-softmax, no diversity loss, no codebook collapse. Just "cluster → predict → re-cluster."
- **Iterative refinement is a reusable idea.** "Bootstrap labels from increasingly good models" applies beyond speech.
- **Softmax-over-cluster is interpretable.** Unlike contrastive, the loss directly trains a classifier — easier to debug.
- **Stepping-stone between wav2vec 2.0 and BEST-RQ.** HuBERT simplified the quantizer (offline k-means vs online learned); BEST-RQ further simplified (frozen random vs offline k-means).

---

## Gotchas & tricks

- **Iteration count matters.** Iter 1's MFCC-clustered labels are noisy; iter 2 and 3 are much better. Skipping to iter 2 or later isn't possible without a model to cluster-on.
- **K-means cluster count.** k=100 is enough for iter 1 (coarse phoneme-like categories); k=500 is enough for later iters. Higher k doesn't help much.
- **Which layer to cluster.** Layer 6 (Base, iter 2), layer 9 (Base, iter 3 for Large). The "optimal layer for clustering" varies with depth; layers in the middle tend to have the best phonetic content. This is an empirical finding.
- **α = 1 (mask-only loss) matters.** With α < 1, the unmasked-position loss dominates and degrades performance; noisy cluster labels hurt more than they help when they're most of the loss.
- **MiniBatchKMeans stability.** With 10K-frame batches and 20 k-means++ restarts, clustering converges reliably. Vanilla KMeans on millions of frames doesn't fit in memory.
- **Fine-tuning with CTC is standard.** Same recipe as wav2vec 2.0.
- **Pretraining compute.** Comparable to wav2vec 2.0 (iter 2 takes ~same as wav2vec 2.0). Iter 3 is additional.
- **BEST-RQ's advantages.** One-shot training (no iterative refinement), random quantizer (no k-means). BEST-RQ matches HuBERT X-Large with much less complexity. This is why post-2022 systems (Llama 3, Kimi k1.5) prefer BEST-RQ.

---

## Sources

- Paper: *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units* — Hsu, Bolte, Tsai, Lakhotia, Salakhutdinov, Mohamed, Facebook AI, 2021, arXiv 2106.07447.
- Paper: *wav2vec 2.0* — Baevski et al., 2020 — the predecessor.
- Paper: *BEST-RQ* — Chiu et al., 2022 — the simpler successor.
- Repo: fairseq HuBERT — https://github.com/pytorch/fairseq/tree/main/examples/hubert.
