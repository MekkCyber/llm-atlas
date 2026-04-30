# Log-Mel Spectrogram

*Depth — the canonical speech input representation: STFT → mel filterbank → log.*

**TL;DR:** Raw 16 kHz waveform is compressed into an **80-channel log-mel spectrogram** at 10 ms stride. STFT with 25 ms Hann window → power spectrogram → triangular mel-filterbank → log. Result: a `[80, num_frames]` tensor where each frame represents 10 ms of audio. This is the input to essentially every modern speech model (Whisper, Conformer, wav2vec2 variants, Llama 3 speech). 10 ms stride is the de-facto standard because it matches human phonetic timescales (phonemes are 50–200 ms, so ≥5 frames per phoneme).

**Prereqs:** basic signal processing
**Related:** [conformer](conformer.md) · [whisper](whisper.md) · [best-rq](best-rq.md)

---

## What it is

A time-frequency representation of audio, designed to approximate human auditory perception. The near-universal input to speech recognition and generation models.

Pipeline:

```
raw 16 kHz waveform → STFT → |·|² → mel filterbank → log → [80, T] log-mel spectrogram
```

---

## How it works

### Step 1: STFT (Short-Time Fourier Transform)

Slide a window across the waveform, apply FFT to each window. Standard parameters (Whisper, Conformer, BEST-RQ):

- **Sample rate**: 16,000 Hz (16 kHz) — the speech-recognition standard.
- **Window length**: 25 ms = 400 samples at 16 kHz.
- **Window function**: Hann window (cosine-tapered; reduces spectral leakage).
- **Hop length (stride)**: 10 ms = 160 samples.

Frames overlap (25 ms window, 10 ms hop → ~60% overlap). Each frame becomes one column of the spectrogram.

Number of frequency bins per frame: for an N-point FFT, `N/2 + 1` bins. Whisper uses `N = 400` → 201 bins. Typical: `N = 512` → 257 bins.

Output: `[num_bins, num_frames]` complex-valued spectrogram.

### Step 2: Power spectrogram

Take the squared magnitude:

```
power[f, t] = |STFT[f, t]|²
```

Real-valued, `[num_bins, num_frames]`.

### Step 3: Mel filterbank

A bank of **triangular filters** with centers spaced on the **mel scale** — approximately log-spaced in Hz above ~1 kHz, linear below. Mimics the frequency-selectivity of the human ear (which is more sensitive to changes at low frequencies than high).

The mel scale:

```
mel(f) = 2595 · log₁₀(1 + f / 700)
```

Inverse:

```
f(m) = 700 · (10^(m / 2595) − 1)
```

For an 80-channel filterbank between 0 Hz and 8000 Hz (Nyquist):
- 82 mel-scale breakpoints (80 filters share inner endpoints).
- Each filter is a triangle peaking at one breakpoint, reaching down to 0 at the adjacent breakpoints.

Apply the filterbank to the power spectrogram:

```
mel_spec[m, t] = Σ_f filter_m[f] · power[f, t]
```

Output: `[80, num_frames]` real-valued, non-negative.

### Step 4: Log

Compress dynamic range (quiet and loud sounds need comparable representation):

```
log_mel[m, t] = log(mel_spec[m, t] + ε)
```

ε is a small constant (1e-10) to avoid log(0). Whisper's specific variant: `log_mel = log10(max(mel_spec, 1e-10))` then clipped to `[max - 8, max]` and rescaled to `[-1, 1]`. Details vary; the principle is the same.

### Output shape

For a 30-second audio clip at 16 kHz:
- 480,000 samples.
- 30 s / 10 ms = 3000 frames.
- 80 mel channels.

Log-mel tensor: `[80, 3000]`.

Whisper chunks into 30-second segments and pads to exactly `[80, 3000]`.

---

## Why these specific parameters

- **16 kHz**: covers up to 8 kHz (Nyquist), which captures all human speech frequency content. Higher sample rates (44.1 kHz CD-quality) add expensive detail not relevant to speech.
- **25 ms window**: long enough to capture one pitch period of adult speech (which is ~5–10 ms), short enough to assume stationarity within the window.
- **10 ms hop**: phonemes last 50–200 ms, so ≥5 frames per phoneme. Gives the ASR model enough temporal resolution.
- **80 mel channels**: empirically optimal for speech. More channels (128, 256) add detail but diminishing returns. Earlier ASR used 40 or 64; modern default is 80.
- **Log scale**: human perception of loudness is approximately logarithmic. Log-mel matches what the ear hears.

### Effective frame rate

At 16 kHz, 10 ms hop → **100 mel frames per second**. Many speech encoders further subsample:
- **Conformer**: 2 conv layers stride 2 each → 4× subsampling → 25 Hz (40 ms per encoded frame).
- **Whisper encoder**: 2 conv layers, second has stride 2 → 2× subsampling → 50 Hz (20 ms per encoded frame).
- **wav2vec 2.0 CNN feature extractor**: 7 conv layers strides (5,2,2,2,2,2,2) → 320× subsampling **from raw waveform** → 50 Hz.

### Alternative input: raw waveform

wav2vec 2.0 skips the mel pipeline entirely and feeds raw 16 kHz samples to a learned CNN that produces features at 20 ms stride. Works well, requires more compute per forward. Most modern pipelines still use log-mel because it's cheap and interpretable.

---

## Why it matters

- **Universal speech input.** Every major speech model (Whisper, Conformer, wav2vec 2.0-Conformer, BEST-RQ, HuBERT-via-CNN, Llama 3 speech) takes log-mel or raw waveform. Understanding log-mel = understanding the speech input layer.
- **Perceptually motivated.** The mel scale and log compression aren't arbitrary — they approximate what the human ear does. Features "matter" in a predictable way.
- **Cheap.** CPU-fast to compute; streaming-friendly (each 10 ms of audio produces one frame).
- **Compatible with any downstream architecture.** CNN, Transformer, Conformer all work on `[80, T]` input.

---

## Gotchas & tricks

- **Windowing matters.** Rectangular window (no taper) has severe spectral leakage. Hann, Hamming, Blackman all work; Hann is the standard.
- **FFT size vs window size.** FFT is zero-padded up to a power of 2 (512, 1024) even if the window is 400 samples. Faster FFT, same information.
- **Low-frequency cutoff.** Some implementations set the lowest mel filter at 20 Hz (avoid DC and low-frequency noise); others start at 0.
- **High-frequency cutoff.** Typically Nyquist (8 kHz at 16 kHz sample rate). Can lower to 7600 Hz to avoid aliased content.
- **Centering.** STFT can be "center" (frame t is centered on sample t · hop) or left-aligned. Librosa uses center by default; some code uses left. Affects boundary behavior.
- **Normalization varies.** Whisper: log10, clip, rescale to [-1, 1]. PyTorch torchaudio: natural log, no normalization. Check compatibility when loading models across frameworks.
- **SpecAugment**: time and frequency masking applied to log-mel spectrograms during training. Frequency masks: zero out `f` consecutive mel bins. Time masks: zero out `t` consecutive frames. Simple, effective data augmentation (Park 2019). Conformer uses it by default.
- **Global vs per-utterance normalization.** Whisper normalizes to a fixed range; others use utterance-level mean/std. Affects train/inference consistency.
- **Streaming vs full.** For streaming ASR, you only have access to audio up to the current time; you can't center-window the latest frame. Streaming implementations shift to left-aligned or zero-pad the right side.
- **80 channels isn't always right.** For music or general audio, 128-256 channels can help. For compressed speech (narrowband telephony, 8 kHz sample rate), 40 channels suffice.
- **MFCCs are a further processing step.** MFCC = DCT of log-mel, keep first ~13 coefficients. Decorrelates mel bands, used by classical ASR. HuBERT's iter-1 uses 39-dim MFCCs (13 + Δ + ΔΔ). Modern deep models use log-mel directly and skip the DCT.

---

## Sources

- Textbook: Rabiner & Juang, *Fundamentals of Speech Recognition*, 1993 — the canonical reference for STFT, mel, and ASR feature extraction.
- Paper: *Whisper: Robust Speech Recognition via Large-Scale Weak Supervision* — Radford et al., 2022 — canonical modern log-mel usage.
- Paper: *Conformer* — Gulati et al., 2020 — log-mel + SpecAugment input.
- Paper: *SpecAugment: A Simple Data Augmentation Method for ASR* — Park et al., 2019, arXiv 1904.08779.
- Paper: *wav2vec 2.0* — Baevski et al., 2020 — raw-waveform alternative.
- Docs: torchaudio.transforms.MelSpectrogram — https://pytorch.org/audio/stable/transforms.html.
