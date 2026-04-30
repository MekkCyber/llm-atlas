# wav2vec 2.0

*Depth — the self-supervised contrastive pretraining that started the "speech BERT" era.*

**TL;DR:** Raw 16 kHz waveform → 7-layer CNN feature extractor (20 ms stride) → Transformer context network. Pretraining masks some time-steps; the model predicts each masked step's **learned quantized target** via a **contrastive loss** over 100 distractors. The quantizer uses **product quantization with Gumbel-softmax** — G=2 groups × V=320 entries → up to 102,400 codewords. After pretraining on LibriLight (~53K hours unlabeled), fine-tune with CTC on as little as **10 minutes of labeled data** to get 4.8 / 8.6 WER on LibriSpeech. Baevski et al. (NeurIPS 2020) — the paper that made self-supervised speech pretraining dominant.

**Prereqs:** [mel-spectrogram](mel-spectrogram.md) (note: wav2vec 2.0 skips this — works on raw waveform), [attention](../../fundamentals/attention.md)
**Related:** [best-rq](best-rq.md) · [hubert](hubert.md) · [conformer](conformer.md)

---

## What it is

Baevski, Zhou, Mohamed, Auli, *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020, arXiv 2006.11477.

The canonical self-supervised speech pretraining method 2020–2022. Before wav2vec 2.0, ASR required hundreds of hours of labeled audio. After: fine-tune a pretrained wav2vec 2.0 on 10 minutes of labeled data and get usable ASR.

Three components:
1. **CNN feature encoder** — raw waveform → 50 Hz latent features. No mel step.
2. **Transformer context network** — adds bidirectional context to the latents.
3. **Quantization module** — maps latents to discrete codes via product quantization with Gumbel-softmax.

Pretraining: mask some latent positions; learn to predict the quantized code at each masked position via contrastive loss.

---

## How it works

### Architecture (Sec. 2)

**CNN feature encoder f : waveform → Z** (Sec. 2, line 88):
- 7 blocks, all 512 channels.
- Strides: (5, 2, 2, 2, 2, 2, 2).
- Kernel widths: (10, 3, 3, 3, 3, 2, 2).
- Output rate: 16000 / (5×2×2×2×2×2×2) = 16000/320 = **50 Hz** (20 ms stride, 25 ms receptive field).

**Transformer context network g : Z → C**:
- Uses **convolutional relative positional embedding** (kernel 128, 16 groups) instead of fixed sinusoidal.
- BASE: 12 layers, d_model 768, FFN 3072, 8 heads.
- LARGE: 24 layers, d_model 1024, FFN 4096, 16 heads.

**Quantization module Z → Q** (Sec. 2, lines 106–123):
- **Product quantization** with **G groups**, **V entries** per group.
- Choose one entry from each group, concatenate → one quantized representation.
- BASE and LARGE: **G = 2, V = 320** → max 102,400 distinct codewords.
- **Gumbel-softmax** (straight-through) for differentiable selection of codebook entry.

### Total loss (Eq. 2)

```
L = L_m + α · L_d
```

- **L_m**: contrastive loss on masked positions.
- **L_d**: codebook diversity loss.
- **α = 0.1** (weight on diversity term).

### Contrastive loss L_m (Eq. 3)

For masked position t:
- `c_t` = context network output at position t.
- `q_t` = true quantized target at position t.
- `Q_t` = {q_t} ∪ **{100 distractors}**: 100 distractors sampled uniformly from other masked positions of the same utterance.

```
L_m = −log [ exp(sim(c_t, q_t)/κ) / Σ_{q̃ ∈ Q_t} exp(sim(c_t, q̃)/κ) ]
```

where sim(a, b) = cosine similarity and temperature κ = **0.1**.

Interpretation: predict the true quantized code at the masked position, discriminating against 100 negative candidates. InfoNCE-style contrastive.

### Diversity loss L_d (Eq. 4)

Prevents codebook collapse (all masked predictions choosing the same code). Maximize entropy of the averaged softmax over each codebook's V entries across a batch:

```
L_d = (1 / (G·V)) · Σ_g [−H(p̄_g)]
    = (1 / (G·V)) · Σ_g Σ_v p̄_{g,v} · log p̄_{g,v}
```

Gumbel-softmax temperature τ is annealed from **2 → 0.5** (BASE) or **2 → 0.1** (LARGE), factor 0.999995/update.

### Masking (Sec. 3.1, Sec. 4.2)

- Sample p = **6.5%** of time-steps as **mask starts**.
- Mask **M = 10 consecutive steps** from each start.
- Result: ~49% of time-steps masked (including overlaps), mean span ≈ 14.7 steps (≈ 300 ms).
- Masked latents are replaced with a **single trained mask embedding**.
- The quantizer sees the **unmasked** features to produce targets — targets are computed from clean input.

### Training data (Sec. 4.1)

- **LS-960** (LibriSpeech 960 h) or **LV-60k** (LibriVox ~53K h; "60k" is loose — actual 53,200 h).
- Fine-tuning settings: 960h, 100h, 10h, 1h, or **10 minutes** of labels.

### Fine-tuning (Sec. 5)

- Add a randomly-initialized linear projection on top of the context net.
- **CTC loss** over characters (26 letters + space + apostrophe + CTC blank).
- **Feature encoder frozen** during fine-tuning.
- The context net and projection train.

### Results — LibriSpeech

LARGE pretrained on LV-60k (~53K h unlabeled), fine-tuned on:

| Labels | test-clean (LM) | test-other (LM) |
|---|---|---|
| 10 minutes | 4.8 | 8.6 |
| 1 hour | 2.9 | 5.8 |
| 10 hours | 2.3 | 4.6 |
| 100 hours | 1.9 | 3.9 |
| 960 hours | **1.8** | **3.3** |

Headline: **1.8 / 3.3 WER on LibriSpeech with 960 h labels** — competitive with much larger supervised baselines. And **4.8 / 8.6 with 10 minutes** — a radical demonstration of self-supervised transfer.

---

## Why it matters

- **The breakthrough paper for self-supervised speech.** Before wav2vec 2.0: hundreds of hours labeled. After: 10 minutes.
- **Raw waveform input.** Skips mel-spectrogram; the CNN feature extractor is trained end-to-end. Philosophical point: human-engineered features aren't needed if you have enough data.
- **Drop-in encoder for any downstream task.** ASR, speaker ID, emotion recognition — all fine-tune well from wav2vec 2.0 pretrained weights.
- **Established the "speech BERT" pattern.** Masked prediction + learned quantizer. HuBERT, w2v-BERT, BEST-RQ are all variants.
- **Public, open weights.** Released by Meta; foundational for every subsequent open speech model.

---

## Gotchas & tricks

- **Gumbel-softmax τ schedule matters.** Starting too cold (τ < 1) causes collapse; starting too hot doesn't discriminate. The 2 → 0.5 schedule is load-bearing.
- **Diversity loss is non-optional.** Without `α · L_d`, the codebook collapses to a handful of codes.
- **Contrastive with 100 distractors is tuned.** Fewer negatives degrades significantly; more is marginal.
- **Masking parameters are load-bearing.** 6.5% starts × span 10 gives ~49% coverage. Less masking = insufficient prediction signal.
- **Raw waveform vs log-mel.** wav2vec 2.0 uses raw; Conformer / Whisper / BEST-RQ use log-mel. Both work; raw has slightly better ceiling, log-mel is cheaper.
- **Feature encoder must be frozen during fine-tune.** Otherwise, unsupervised features drift and hurt downstream performance.
- **CTC decoder is a separate concern.** CTC with a 4-gram language model on top significantly improves WER; without LM is ~1-2 points worse.
- **XLS-R is multilingual wav2vec 2.0.** Same architecture, trained on 436K hours across 128 languages. Good starting point for low-resource-language ASR.
- **wav2vec 2.0 is harder to train than BEST-RQ.** Gumbel-softmax schedule, diversity loss, codebook collapse — lots of failure modes. BEST-RQ's frozen-random quantizer sidesteps most of them, which is why Llama 3 and Kimi prefer BEST-RQ.
- **Large context benefits.** Pretraining on longer utterances (30+ s) helps downstream long-form ASR. Standard pretraining uses crops of ~15 s.

---

## Sources

- Paper: *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations* — Baevski, Zhou, Mohamed, Auli, Facebook AI, NeurIPS 2020, arXiv 2006.11477.
- Paper: *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale* — Babu et al., 2021, arXiv 2111.09296 — multilingual variant.
- Paper: *HuBERT* — Hsu et al., 2021 — alternative with k-means labels.
- Paper: *BEST-RQ* — Chiu et al., 2022 — frozen-random-projection alternative.
- Repo: fairseq wav2vec 2.0 — https://github.com/pytorch/fairseq/tree/main/examples/wav2vec.
