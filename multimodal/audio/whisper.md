# Whisper

*Depth — OpenAI's 680K-hour-trained encoder-decoder for ASR, translation, language ID, and VAD.*

**TL;DR:** An encoder-decoder Transformer trained on **680,000 hours of weakly-supervised multilingual audio** scraped from the web. Input: 80-channel log-mel spectrogram of a 30-second chunk. Encoder: 2 conv layers (stride 2 in the second) + sinusoidal PE + Transformer blocks. Decoder: standard Transformer decoder with learned PE. **Multi-task via special tokens**: the decoder sequence starts with `<|startoftranscript|>`, a language token (99 supported), a task token (transcribe/translate), a timestamp flag, then the transcript. Sizes: tiny (39M) to large (1.55B). The most-used open speech system since late 2022. Radford et al. (OpenAI, Dec 2022).

**Prereqs:** [mel-spectrogram](mel-spectrogram.md), [transformer-block](../../architectures/transformer-block.md)
**Related:** [conformer](conformer.md) · [best-rq](best-rq.md)

---

## What it is

Radford, Kim, Xu, Brockman, McLeavey, Sutskever, *Robust Speech Recognition via Large-Scale Weak Supervision*, arXiv 2212.04356.

An end-to-end multi-task speech model. Unlike prior ASR (which was trained task-specifically, often with hand-tuned language models or phoneme lexicons), Whisper is a single Transformer that does:

- **ASR** (automatic speech recognition) — 96+ languages.
- **Translation** — from any of 96+ languages into English.
- **Language ID** — identify the spoken language.
- **Voice activity detection (VAD)** — detect silent vs speech segments.

All via special tokens in the decoder sequence — no task-specific heads.

---

## How it works

### Input features (Sec. 2.2)

- **Sample rate**: 16,000 Hz (resampled if input is different).
- **Features**: 80-channel log-magnitude **mel spectrogram**.
- **STFT window**: 25 ms, **stride 10 ms**.
- **Normalization**: globally scaled to `[-1, 1]` with approximately zero mean across the pretraining dataset (specific normalization: log10, clip to [max-8, max], rescale).
- **Chunking**: 30-second segments. Audio shorter than 30s is padded; longer is split.

### Encoder

```
[80, 3000] log-mel input (30 s × 100 Hz)
    ↓
Conv1D(80 → d_model, kernel=3, stride=1), GELU    # input stem
    ↓
Conv1D(d_model → d_model, kernel=3, stride=2), GELU  # subsampling: 100 Hz → 50 Hz
    ↓
+ sinusoidal positional encoding
    ↓
Transformer encoder blocks (pre-activation residual)
    ↓
LayerNorm
    ↓
encoder output: [d_model, 1500] (50 Hz × 30 s)
```

Key details:
- **Two conv layers** at the bottom. The second has stride 2 → **2× temporal subsampling**. Encoder operates at ~50 Hz (20 ms per frame).
- **Sinusoidal positional embeddings** (not learned, not rotary) — added after the stem.
- **Pre-activation residual** (LN before attn/FFN): Child et al. 2019 style.
- **Final LayerNorm** on encoder output.

### Decoder

```
[prefix tokens: SOT, lang, task, timestamps_flag, text...]
    ↓
Token embeddings (tied with output)
+ learned positional embeddings
    ↓
Transformer decoder blocks:
    - masked self-attention over decoder tokens
    - cross-attention over encoder output
    - FFN
    ↓
LayerNorm
    ↓
Output projection (tied with input embeddings)
    ↓
next-token logits
```

Key details:
- **Learned positional embeddings** (not sinusoidal — differs from encoder).
- **Tied input-output embeddings** (Press & Wolf 2017).
- Same width and number of blocks as the encoder.

### Tokenizer (Sec. 2.2)

- **Byte-level BPE**, same as GPT-2 for English-only models.
- Multilingual models: **re-fit** on multilingual text (same vocab size, different merges) to avoid over-fragmenting non-English.

### Multi-task format (Sec. 2.3, Figure 1)

A single decoder sequence encodes the task via special tokens:

```
<|startoftranscript|> <|language|> <|task|> <|notimestamps|> text tokens <|endoftranscript|>
```

Components:

1. `<|startoftranscript|>` — always first.
2. **Language token** — one of 99 language codes (e.g., `<|en|>`, `<|de|>`) or `<|nospeech|>` (VAD: no speech detected).
3. **Task token** — `<|transcribe|>` (recognize in source language) or `<|translate|>` (recognize and translate to English).
4. **Timestamp flag** — `<|notimestamps|>` disables timestamps; omitting it causes the model to emit timestamps inline.
5. **Transcript tokens**, with timestamp tokens interleaved if in timestamp mode. Timestamps quantized to **20 ms** (matches encoder frame rate).
6. `<|endoftranscript|>` — end of sequence.

With some probability during training, the **preceding transcript** is prepended to the decoder context (for continuity across 30-s chunks); loss masked over that prefix.

### Training data (Sec. 1)

- **680,000 hours** of weakly-supervised audio.
- **117,000 hours** covering 96 non-English languages.
- **125,000 hours** of X→English translation.
- Rest (~438K h): English ASR.
- Data scraped from web audio with captions/subtitles; de-duplicated against evaluation sets.

"Weakly supervised" = the labels are whatever captions/subtitles came with the audio. Noisy; captures the distribution of the web.

### Model family (Table 1)

| Model | Layers | Width | Heads | Params |
|---|---|---|---|---|
| Tiny | 4 | 384 | 6 | 39M |
| Base | 6 | 512 | 8 | 74M |
| Small | 12 | 768 | 12 | 244M |
| Medium | 24 | 1024 | 16 | 769M |
| Large | 32 | 1280 | 20 | 1550M |

Later variants: **Large-V2** (2.5× more epochs, SpecAugment, stochastic depth, BPE dropout), **Large-V3** (further data improvements).

### Training

- AdamW with gradient-norm clipping.
- Linear LR decay to zero after 2048-step warmup.
- Batch 256 segments.
- 2^20 updates (≈ 2–3 epochs over 680K hours).
- FP16 with dynamic loss scaling + activation checkpointing.
- No data augmentation or regularization in the original run.

---

## Why it matters

- **The most-used open speech system since late 2022.** ChatGPT's voice feature, Descript, Fireflies, Otter — the entire transcription ecosystem built on Whisper or derivatives.
- **Robustness.** 680K hours of noisy web data gives strong handling of accents, background noise, and low-quality recordings. Beats cleaner-trained models on noisy real-world audio.
- **Zero-shot multilingual.** One model handles 99 languages. You don't need to pick a language at load time; Whisper detects it.
- **Timestamps by default.** Word-level timestamps come for free via the 20-ms timestamp tokens. Useful for subtitles, search-by-time.
- **Multi-task via prompting.** The special-token task-control pattern has influenced subsequent speech models and is a clean design.

---

## Gotchas & tricks

- **30-second chunks.** The model is trained on exactly 30-s windows. Longer audio must be chunked and re-stitched. Word boundaries near chunk edges can be misaligned.
- **Hallucinations on silence.** Whisper can hallucinate text during silent or near-silent audio. The `<|nospeech|>` token partly mitigates this, but hallucinations still occur. VAD preprocessing is the common workaround.
- **Language detection is imperfect.** If the first ~30 s of audio is silence or music, language detection confidence is low. Sometimes picks wrong language.
- **Large-V2 and Large-V3 are mandatory for production.** The original Large (V1) has known quality issues; V2 (better training) and V3 (improved data) are the go-to checkpoints.
- **Translation is only X → English.** Whisper translates *to English*, not between arbitrary languages. Multilingual translation requires an LLM post-hoc.
- **Timestamp resolution is 20 ms.** The quantization matches the encoder frame rate. Finer timestamps require post-processing alignment (e.g., Forced Aligner on top).
- **English-only models exist.** Whisper releases include `.en` variants (tiny.en, base.en, small.en, medium.en) trained only on English. ~1% better than multilingual counterparts on English tasks.
- **Prompting affects output style.** Passing an **initial prompt** (e.g., proper-noun spellings, domain context) biases the decoder toward those styles. Useful for domain-specific transcription.
- **Beam search + temperature fallback.** Default decoding uses beam search with beam=5; if confidence is low (compression ratio, log probability), falls back to sampling with temperature 0.2–1.0. Important for robustness on noisy inputs.
- **VAD preprocessing.** Silero VAD or py-webrtcvad on the raw waveform, then feed only speech segments to Whisper. Reduces hallucinations and compute.
- **Distilled variants.** Distil-Whisper (Hugging Face 2023) is a ~2× faster, ~6× smaller distilled Whisper. Useful for CPU/edge deployment.
- **Streaming Whisper.** Original Whisper is not streaming-friendly (needs 30-s chunks). Streaming variants (FasterWhisper, whisper.cpp with streaming) use small overlapping windows. Accuracy degrades slightly vs full 30 s.
- **Fine-tuning.** Whisper fine-tunes well on domain-specific audio with LoRA or full fine-tune. Hugging Face transformers has canonical examples.

---

## Sources

- Paper: *Robust Speech Recognition via Large-Scale Weak Supervision* — Radford, Kim, Xu, Brockman, McLeavey, Sutskever (OpenAI), Dec 2022, arXiv 2212.04356.
- Repo: https://github.com/openai/whisper.
- Paper: *Distil-Whisper* — Gandhi et al., HF, 2023, arXiv 2311.00430.
- Repo: faster-whisper — https://github.com/guillaumekln/faster-whisper — CTranslate2-based faster inference.
- Repo: whisper.cpp — https://github.com/ggerganov/whisper.cpp — C++ port with streaming.
