# Conformer

*Depth — convolution-augmented Transformer; the de-facto speech-encoder architecture since 2020.*

**TL;DR:** A Transformer block with a **convolution module** added: FFN → Multi-head Self-Attention → **Convolution** → FFN, with `½` residual weights on the FFNs (the "Macaron-Net" trick). Convolution captures local features (phoneme-scale), attention captures global context. Attention uses **relative positional encoding**. Conformer-L (118M params) hit 2.1 / 4.3 WER on LibriSpeech test-clean / test-other (SOTA at release). Used by Llama 3's speech encoder (1B variant) and most modern ASR systems. Introduced by Gulati et al. (Interspeech 2020).

**Prereqs:** [attention](../../fundamentals/attention.md), [transformer-block](../../architectures/transformer-block.md), [mel-spectrogram](mel-spectrogram.md)
**Related:** [whisper](whisper.md) · [best-rq](best-rq.md) · [wav2vec2](wav2vec2.md)

---

## What it is

Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*, Interspeech 2020, arXiv 2005.08100.

A Transformer block modified for speech. Observation: pure Transformers attend globally but lack the **local phoneme-scale inductive bias** that convolutions provide. Pure CNNs have local bias but weak long-range context. Conformer combines both.

Since 2020, Conformer or Conformer-derived blocks have been the standard speech encoder. Used by wav2vec 2.0 Conformer, BEST-RQ (Kimi k1.5, Llama 3), Whisper-style models, and most modern ASR.

---

## How it works

### The Conformer block (Sec. 2.4, Eq. 1)

For input `x_i` to block i:

```
x̃_i  = x_i   + ½ · FFN(x_i)        ← first half-step FFN
x'_i = x̃_i  + MHSA(x̃_i)            ← multi-head self-attention with rel PE
x''_i= x'_i  + Conv(x'_i)           ← NEW: convolution module
y_i  = LayerNorm(x''_i + ½ · FFN(x''_i))  ← second half-step FFN + LN
```

The two `½ · FFN` residuals are "**Macaron-Net**" half-step FFNs (Lu 2019). Splitting one full-step FFN into two half-step FFNs at the start and end of the block. Modest improvement over one full-step.

### Multi-Head Self-Attention module

Pre-norm residual unit:

```
x → LayerNorm → MHSA_relpe → Dropout → +
```

**Relative positional encoding** from Transformer-XL (Dai 2019) — makes the model length-robust. Important because speech utterances vary in length.

### Convolution module (Sec. 2.2, Figure 2)

The novel piece. Structure:

```
x → LayerNorm                        (pre-norm)
  → PointwiseConv (expand 2×)        (channel expansion)
  → GLU                              (halves channels back via gating)
  → DepthwiseConv (kernel size 32)   (local 1-D temporal conv along time)
  → BatchNorm                        (stabilizes)
  → Swish                            (activation)
  → PointwiseConv                    (project back)
  → Dropout → + (residual)
```

Key design choices:
- **Pointwise + depthwise**: factorized conv, similar to MobileNet. Much cheaper than a full conv.
- **GLU (Gated Linear Unit)**: `GLU(x) = x_a ⊙ σ(x_b)` where x is split in half. Adds a gating nonlinearity.
- **Depthwise kernel 32**: each input channel is convolved independently with a 32-tap filter. Captures ~320 ms of local context at 10 ms stride (or ~640 ms after 2× subsampling).
- **BatchNorm** inside the block — unusual for Transformers (which normally use LayerNorm). Conformer uses BatchNorm specifically here.
- **Swish** activation (SiLU): `Swish(x) = x · σ(x)`. Smoother than ReLU.

### Feed-Forward module (Sec. 2.3, Figure 4)

Pre-norm unit with **expansion factor 4**:

```
x → LayerNorm
  → Linear (D → 4D)
  → Swish
  → Dropout
  → Linear (4D → D)
  → Dropout
  → + (residual)
```

Applied with `½` residual weight (the Macaron trick).

### Variants (Table 1)

| Hyperparam | Conformer-S | Conformer-M | Conformer-L |
|---|---|---|---|
| Params | **10.3M** | **30.7M** | **118.8M** |
| Encoder layers | 16 | 16 | 17 |
| Encoder dim | 144 | 256 | 512 |
| Attention heads | 4 | 4 | 8 |
| Conv kernel | 32 | 32 | 32 |
| Decoder layers | 1 (LSTM) | 1 (LSTM) | 1 (LSTM) |
| Decoder dim | 320 | 640 | 640 |

The decoder is a single-LSTM-layer RNN-Transducer in all Conformer variants — not a Transformer decoder. The encoder does the heavy lifting.

### Input (Sec. 3.1)

- **80-channel log-mel filterbank**, 25 ms window, **10 ms stride**.
- **SpecAugment** with F = 27, 10 time masks, pS = 0.05.

### Training hyperparameters (Sec. 3.1)

- Dropout 0.1 in each residual unit.
- ℓ2 regularization 1e-6.
- Adam: β₁=0.9, β₂=0.98, ε=1e-9.
- Transformer LR schedule with 10K warmup, peak LR = 0.05/√d.

### Results on LibriSpeech (Table 2)

| Model | Params | test-clean (no LM) | test-other (no LM) | test-clean (LM) | test-other (LM) |
|---|---|---|---|---|---|
| Conformer-S | 10.3M | 2.7 | 6.3 | 2.1 | 5.0 |
| Conformer-M | 30.7M | 2.3 | 5.0 | 2.0 | 4.3 |
| **Conformer-L** | 118.8M | **2.1** | **4.3** | **1.9** | **3.9** |

Conformer-L was SOTA at release — 15% relative improvement on test-other over the prior Transformer Transducer (139M, 2.0/4.6 with LM).

---

## Why it matters

- **Default speech encoder architecture since 2020.** Almost every modern ASR and SSL speech model uses Conformer blocks.
- **Combines local + global.** Convolution catches phoneme-scale features; attention catches utterance-scale context. The combination empirically beats either alone.
- **Well-defined, well-tuned.** Five years of hyperparameter refinement → the default config just works. Plug-and-play.
- **Scales smoothly.** Used at 10M (Conformer-S) and 1B+ (BEST-RQ, Llama 3 speech encoder). Same building block.
- **Pairs with any training recipe.** Supervised CTC/RNN-T (original), self-supervised contrastive (wav2vec 2.0 Conformer), self-supervised masked-prediction (BEST-RQ), weakly-supervised (Whisper).

---

## Gotchas & tricks

- **BatchNorm in the conv module** is non-standard. It works well in speech (where utterance-length batches are natural) but can be tricky at inference if batch stats aren't well-calibrated. Some implementations use GroupNorm or LayerNorm instead.
- **Kernel size 32** is the standard; some variants use 15 or 31. Bigger kernel = more local context, higher cost.
- **Relative PE is essential.** Absolute PE makes Conformer much worse at variable-length utterances.
- **Macaron-FFN ½-residual is subtle.** Don't forget the ½ factor — without it, the two FFNs add up to double-FFN-residual and destabilize training.
- **Depthwise convolution is cheap.** Despite kernel size 32, depthwise is O(channels × time) — much less than a full `channels² × time` 1D conv.
- **Subsampling stem.** Standard pipeline: 2 conv layers at the bottom of the encoder, each stride 2, reducing frame rate from 100 Hz (log-mel at 10 ms) to 25 Hz. Cuts the sequence length 4× before the expensive Conformer blocks.
- **Composes with Flash Attention.** The MHSA inside Conformer is standard and can use FA2/FA3 kernels.
- **LSTM decoder vs Transformer decoder.** Conformer's original paper uses an LSTM decoder (for RNN-T). Whisper and modern systems use Transformer decoders instead. The Conformer architecture refers to the **encoder only**.
- **GLU vs Swish.** GLU in the conv module, Swish in the FFN. Not interchangeable; chosen empirically.

---

## Sources

- Paper: *Conformer: Convolution-augmented Transformer for Speech Recognition* — Gulati, Qin, Chiu, Parmar, Zhang, Yu, Han, Wang, Zhang, Wu, Pang, Google, Interspeech 2020, arXiv 2005.08100.
- Paper: *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View (Macaron-Net)* — Lu et al., ICLR 2020, arXiv 1906.02762 — origin of the ½-residual FFN trick.
- Paper: *Transformer-XL* — Dai et al., 2019 — relative positional encoding.
- Paper: *SpecAugment* — Park et al., 2019, arXiv 1904.08779.
- Code: ESPnet, NeMo, Lightning, torchaudio — all have reference Conformer implementations.
