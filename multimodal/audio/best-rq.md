# BEST-RQ

*Depth — self-supervised speech pretraining with a frozen random-projection quantizer.*

**TL;DR:** Same masked-prediction pretraining idea as wav2vec 2.0 / HuBERT, but the **quantizer is a frozen random projection + nearest-neighbor codebook lookup** — not a learned codebook, not iterative k-means. Labels come from projecting input mel features through a random matrix and looking up the closest entry in a random codebook of 8192 vectors. Despite using random (not optimized) labels, BEST-RQ matches or beats wav2vec 2.0 / HuBERT / w2v-BERT on LibriSpeech. Adopted by Llama 3 and Kimi k1.5 for speech encoder pretraining because it's simpler and more stable. Chiu et al. (ICML 2022).

**Prereqs:** [mel-spectrogram](mel-spectrogram.md), [conformer](conformer.md)
**Related:** [whisper](whisper.md) · [wav2vec2](wav2vec2.md) · [hubert](hubert.md)

---

## What it is

Chiu et al., *Self-supervised Learning with Random-projection Quantizer for Speech Recognition*, ICML 2022, arXiv 2202.01855.

A radically simpler alternative to wav2vec 2.0's learned-codebook quantization and HuBERT's k-means pseudo-labels. Observation: for masked-prediction self-supervised pretraining, **the quantizer doesn't have to be optimal** — it just has to be consistent (same input → same label) and diverse enough to force the encoder to learn contextual structure.

Replace the learned quantizer with a **frozen random projection + nearest-neighbor lookup**. Dramatically simpler, works as well or better.

---

## How it works

### The quantizer (Sec. 3.1, Eq. 1)

For input frame-vector `x ∈ ℝ^d`:

```
y = argmin_i  ||L2normalize(c_i) − L2normalize(A·x)||
```

Components:
- **A ∈ ℝ^(h × d)**: projection matrix. Initialized with **Xavier init** (Glorot). **Frozen** after init.
- **C = {c_1, ..., c_n} ∈ ℝ^h × n**: codebook of n vectors. Initialized from **standard normal**. **Frozen**.
- Both `A·x` and `c_i` are **L2-normalized** before the nearest-neighbor lookup (equivalent to cosine nearest neighbor on the projected vector).
- Input `x` is also normalized to **zero mean, unit variance** before projection — "critical" per the paper.

Key parameters (Sec. 4.1.1, LibriSpeech non-streaming):
- **n = 8192** codebook entries.
- **h = 16** codebook dimensionality.
- Input: **80-dim log-mel**, stacked in groups of 4 (because encoder subsamples 4×) → 320-dim stacked features projected to 16.

### Masking (Sec. 3)

Same pattern as wav2vec 2.0 / HuBERT:
- Mask applied **directly on input speech** (log-mel frames), not on encoder output.
- Per-frame Bernoulli mask-start with probability `p`.
- Each mask covers a fixed-length span.
- LibriSpeech non-streaming: mask length **400 ms**, probability **0.01**.
- Masked frames replaced with noise ~ N(0, 0.1²).

### Objective (Sec. 3.2)

BERT-style masked prediction:
- Encoder produces a representation for every frame.
- **Softmax head** predicts the quantized code for each masked position.
- **Cross-entropy loss** on masked positions only.

```python
# Pretraining forward (per masked position)
z = encoder(masked_input)                 # [T, d_encoder]
target_code = quantizer(original_input)   # [T, integer in 0..n-1]  (computed once, frozen)
logits = softmax_head(z)                  # [T, n]
loss = cross_entropy(logits[masked], target_code[masked])
```

### Why random works (Sec. 3.4)

The paper's justification: **the quantization quality doesn't need to be good**. What matters for self-supervised pretraining is that:

1. The quantizer produces **consistent** discrete labels — same input x → same code.
2. The codes are **diverse enough** to be informative (not all the same code).

A random projection roughly preserves the distribution of the speech data (Johnson-Lindenstrauss intuition). The random codebook gives an approximate discretization. Consistent and diverse — that's sufficient.

With enough unsupervised data, the gap in quantization quality vs a learned quantizer becomes negligible.

Additional benefits of frozen random:
- **No Gumbel-softmax temperature schedule** (wav2vec 2.0 requires annealing from 2 → 0.1).
- **No codebook collapse** (learned quantizers sometimes collapse to a single code).
- **No diversity loss** (wav2vec 2.0 adds an entropy term over codebook usage).
- **No iterative refinement** (HuBERT needs 2–3 iterations with re-clustering).

Just initialize, freeze, pretrain. Simple.

### Encoder (Sec. 3.2, 4.1.1)

- **Conformer-based** (Gulati 2020, see [conformer](conformer.md)).
- LibriSpeech config: 2 conv layers at the bottom (4× subsampling), then **24 Conformer layers**, **0.6B params**.

### Results — LibriSpeech (Table 1)

At 0.6B params:

| Method | test-clean / test-other (no LM) | test-clean / test-other (LM) |
|---|---|---|
| wav2vec 2.0 | 2.2 / 4.5 | 1.8 / 3.3 |
| HuBERT Large | — | 1.9 / 3.3 |
| HuBERT X-Large | — | 1.8 / 2.9 |
| w2v-Conformer XL | 1.7 / 3.5 | 1.5 / 3.2 |
| w2v-BERT XL | 1.5 / 2.9 | 1.5 / 2.8 |
| **BEST-RQ** | **1.6 / 2.9** | **1.5 / 2.7** |

Matches or beats learned-quantizer methods at the same scale. With less complexity.

### Streaming variant (Sec. 4.1.2)

For streaming ASR: mask length 300 ms, probability 0.02, 2-frame stacking instead of 4. BEST-RQ's streaming results beat wav2vec 2.0 streaming on both WER and latency (Table 2).

---

## Why it matters

- **Simpler than wav2vec 2.0 / HuBERT.** No learned codebook, no Gumbel-softmax, no iterative clustering. Just a random matrix.
- **Matches SoTA.** Competitive with w2v-BERT on LibriSpeech at the same scale.
- **Adopted by production systems.** Llama 3 speech encoder (see [cross-attention-adapter](../vision/cross-attention-adapter.md) for Llama 3's overall multimodal recipe) and Kimi k1.5 both use BEST-RQ for speech pretraining.
- **Stable training.** No known collapse failure modes. Easier to reproduce.
- **Reusable insight.** The "random projection is sufficient for self-supervised targets" observation applies beyond speech — useful anywhere you need pseudo-labels for masked prediction.

---

## Gotchas & tricks

- **Input normalization is critical.** Without zero-mean-unit-variance normalization of the input, the random projection collapses to using only a small subset of codes. The paper emphasizes this.
- **Frame stacking.** Input features are stacked to match the encoder's subsampled frame rate. 4× for non-streaming (80 mel × 4 → 320 → 16), 2× for streaming.
- **Codebook size 8192.** Not heavily optimized — the paper shows it's robust to 2048–16384. Below 2048 loses diversity; above 16384 is unnecessary.
- **Codebook dimensionality 16.** Similarly robust. The projection from 320 to 16 gives the quantizer its discretization; higher dim = finer codebook but not much benefit.
- **Xavier init of A.** Could also use random normal or orthogonal; Xavier gives slightly better results empirically.
- **Pretraining compute.** Similar to wav2vec 2.0 / HuBERT — 0.6B params × millions of unlabeled audio hours. Multi-GPU, multi-day runs.
- **Fine-tuning**: add a CTC head on top of encoder; freeze encoder or fine-tune depending on labeled-data size.
- **Incompatible with wav2vec 2.0 checkpoints.** Different pretraining target; can't warm-start one from the other.
- **No diversity loss needed.** The random codebook's diversity is inherent (it's random); no auxiliary entropy term required.

---

## Sources

- Paper: *Self-supervised Learning with Random-projection Quantizer for Speech Recognition* — Chiu, Qin, Zhang, Han, Wu, Google, ICML 2022, arXiv 2202.01855.
- Paper: *wav2vec 2.0* — Baevski et al., NeurIPS 2020, arXiv 2006.11477 — the predecessor with learned codebook.
- Paper: *HuBERT* — Hsu et al., 2021, arXiv 2106.07447 — the iterative-k-means alternative.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, Sec. 8 — uses BEST-RQ for speech encoder pretraining (on a 1B Conformer variant).
- Paper: *Kimi k1.5* — Moonshot AI, 2025 — also uses BEST-RQ.
