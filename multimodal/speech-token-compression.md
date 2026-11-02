# Speech Token Compression (CoSTA)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-hoc technique for spoken dialogue models that *decouples* the token space the LM generates from the token space the vocoder consumes. The LM is adapted to produce a compressed token stream, which a deterministic expander maps back to the original speech tokens before synthesis. The audio tokenizer and vocoder are untouched. ~1.7× faster speech-token generation on bilingual benchmarks, preserved response quality.

**Prereqs:** [../fundamentals/_tokenization](../fundamentals/_tokenization.md)
**Related:** [../fundamentals/bpe](../fundamentals/bpe.md)

---

## What it is

Modern spoken dialogue models (SpeechGPT-style, Qwen2.5-Audio) have the LM emit *speech tokens* at the native rate of the audio tokenizer — typically 25–75 Hz. The vocoder consumes those tokens and renders waveform. Latency is dominated by the number of tokens the LM must emit per second of audio, which is set by the audio tokenizer's rate.

CoSTA breaks the tie: train the LM to emit *fewer, denser* tokens from a compressed vocabulary, and interpose a deterministic expander that reinflates them to the original speech-token stream for the vocoder.

## How it works

### Vocabulary design

Build a compressed vocabulary $V_c$ by merging frequently-cooccurring speech-token n-grams in the training speech corpus (BPE-style, but on speech tokens). Each compressed token in $V_c$ maps to a fixed sequence of 1–$k$ original tokens.

### Adaptation (post-hoc)

Starting from a trained spoken-dialogue LM:

1. Replace the LM output head to predict $V_c$ instead of the original vocabulary.
2. Retokenize the training corpus with the compressed vocabulary.
3. Fine-tune briefly on dialogue data under the compressed vocab.

The audio tokenizer and vocoder are not touched.

### Inference

- LM emits compressed tokens at a lower rate.
- Expander (a lookup table) maps each compressed token to its original-token sequence.
- Vocoder renders waveform from the expanded sequence as usual.

Deterministic expansion keeps vocoder behavior and voice identity unchanged.

## Why it matters

- Spoken dialogue latency is set by the LM's token rate; cutting that rate by ~40% translates directly to lower first-byte latency and higher throughput.
- Vocoder quality and voice identity depend on the original tokenizer; by leaving both untouched the technique composes with any existing speech stack.
- Shows that the LM-side token space can be optimized *independently* from the signal-processing side — a general lever for any model whose output tokens feed a downstream deterministic decoder (image patches, video frames).

## Gotchas & tricks

- Compression ratio is bounded by audio-token entropy. In speech, long silence/filler tokens compress very well; dense content words less so. Expect larger speedups on casual dialogue than on dense technical speech.
- Rare original tokens may not get a dedicated compressed token; keep an "escape" symbol so the LM can emit uncompressed when needed.
- The LM must learn the compressed vocabulary quickly or you lose capability during fine-tuning; a brief, careful SFT phase matters.
- Transfers across backbones but the compressed vocabulary itself is tokenizer-specific — re-derive $V_c$ when switching speech tokenizers.
- Compatible with low-bitrate speech tokenizers (fewer bits per original token) for compound savings.

## Sources

- Paper: *CoSTA: Compressed Speech Token Adaptation for Efficient Spoken Dialogue Models* — Deng, Li, Liu, Peng, Xie, Song, Ren, Zhang, IACAS / UCAS / Wuhan AI Research, 2026 — [arXiv:2610.21143](https://arxiv.org/abs/2610.21143).
- Related: *SpeechTokenizer* and *EnCodec* — the audio tokenizers whose output stream CoSTA compresses post-hoc.
