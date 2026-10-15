# Case Study: Fish Audio Drama 3

*A 35B total / 3B active MoE speech generation model that turns text into *directed* performances. The paper matters because it bundles together a hybrid-attention speech backbone, a codebook-MTP audio decoder, a learned preference reward RL stage, and a prefill–decode-disaggregated serving stack — any one of which would be a solid individual contribution. Streams at **271 fps on one H200** (vs. 177 for Fish Audio S2), despite being 7× larger.*

**Related concepts:** [../architectures/_moe.md](../architectures/_moe.md) · [../multimodal/codebook-mtp-decoder.md](../multimodal/codebook-mtp-decoder.md) · [../pre-training/mtp.md](../pre-training/mtp.md) · [../post-training/grpo.md](../post-training/grpo.md) · [../post-training/_rewards.md](../post-training/_rewards.md) · [../quantization/fp8.md](../quantization/fp8.md) · [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What this is

Fish Audio Drama 3 (October 2026) is an open-source speech generation model that treats TTS as a **directed performance** rather than a text → waveform mapping. It clones voices from one or more reference clips, follows natural-language direction (point tags like `[laughs]`, scoped tags like `<whisper>...</whisper>`, free-form director's notes), revises takes across multi-turn conversations, voices multi-character scenes, edits speech via infilling, and streams long-form output.

The paper is interesting for *how* it does these things in one model: a **hybrid linear-attention backbone** (Gated DeltaNet + gated full attention) inherits the compute profile of a modern reasoning LLM, a **codebook-MTP audio decoder** emits 10 codebooks per frame per the residual-codec recipe, and a **GRPO RL stage** with a learned preference RM polishes direction-following and expressivity.

---

## Architecture at a glance

```
Backbone (initialized from Qwen3.5-35B-A3B)
  ├─ 40 transformer layers
  ├─ block pattern: 3 × Gated-DeltaNet  +  1 × gated full attention
  │     (so ~1 in 4 layers maintains a KV cache)
  ├─ MoE FFN at every layer
  │     256 routed experts, top-8 per token
  │     + 1 shared expert
  │     ≈ 3B of 35B params active per token
  └─ RoPE positional encoding

Audio decoder (codebook-MTP; 13 layers total)
  ├─ 4 shared transformer blocks (gated full attention + SwiGLU, backbone width)
  └─ 9 per-codebook blocks, one per residual codebook c_1 … c_9
        each takes previous-layer output + embedding of already-emitted c_0..c_{i-1}
        each has its own weights and output head
        causal attention over the whole sequence (streams)

Partial fusion
  Only c_0 (semantic) and c_1 are summed back into the backbone,
  scaled by 1/√2. c_2..c_9 flow only to the decoder.
```

Hidden size, head counts, and expert intermediate size are not disclosed in the report.

Depth file for the audio decoder: [../multimodal/codebook-mtp-decoder.md](../multimodal/codebook-mtp-decoder.md).

---

## Audio codec

The decoder emits tokens for a **frozen DAC-based codec** (reused from Fish Audio S2):

- **44.1 kHz** sample rate, **~21.5 frames per second**.
- **10 codebooks per frame.** `c_0` is a *semantic* code (4,096-token vocabulary added to the LM vocabulary); `c_1..c_9` are residual *acoustic* codebooks.
- **Fully causal encoder and decoder** — both sides stream.

Per-codebook residual vocabulary size is not stated.

---

## Training recipe

### Pre-training (three stages)

- **Stages 1–2.** Sequences up to **8K tokens**, mix of text documents and multi-turn TTS conversations. Stage 2 **doubles the batch** and continues without LR decay. Data: transcribed speech (quality-controlled) + text, several hundred billion tokens, **~80 languages and dialects**.
- **Stage 3.** Sequences up to **16K tokens**, adds whole-recording and line-by-line long-form synthesis, streaming-interleaved input, and infilling.
- **Row packing** at **128K tokens** with document masking.
- **Optimizer:** Muon.

### Loss

```
L = CE_text + 0.5 · CE_semantic(c_0) + 0.5 · mean_{i≥1} CE_codebook(c_i) + aux-balance
```

The aux load-balancing loss (standard MoE) is active during pre-training and SFT.

### Instruction SFT

One stage from the Stage-3 checkpoint, sequences up to **32K tokens**. Covers direction tags, style prefixes, multi-turn director's notes, multi-speaker dialogue, and whole-scene dubbing. Recipe selection used **held-out NLL of semantic tokens** rather than training loss.

### RL

Reinforcement learning on the backbone only — the decoder is frozen.

- **Algorithm:** GRPO (Dr. GRPO mean-centered advantages), **no KL penalty**, **no reference model**. PPO-style ratio with **asymmetric clip** `ε_low = 0.2, ε_high = 0.28`.
- **Sampling:** 8 takes per prompt, temperature 1, no top-p / top-k. Length capped at ~2.5× expected duration and 4,096 frames (~190 s).
- **Group filter:** drop groups whose reward std < 0.05 (uninformative).
- **Batch:** 1,536 prompts × 8 takes per step = 12,288 takes. Constant Muon LR, no weight decay, few steps.
- **Reward:** a single smaller **learned preference RM** (a speech LM of the authors' own), trained with Bradley–Terry loss + label smoothing on human preference pairs. Scores standardized against a frozen anchor, clipped to ±3, converted to expected win rate within each group.

---

## Capability-specific designs

- **Voice cloning.** References go in the system turn (one reference) or as numbered reference blocks with speaker-tagged transcripts (multi-reference). **No separate audio encoder** — references are consumed as codec tokens.
- **Direction following.** Point tags (`[laughs]`), scoped tags (`<whisper>…</whisper>`), style prefixes, free-form director's notes. Multi-turn revision keeps earlier takes as audio context.
- **Infilling.** Regenerate an edit window given the full target text and the source audio tokens before and after the window; the whole sequence is then decoded.
- **Streaming.** Streaming-interleaved text input is trained in Stage 3; the fully causal codec + causal decoder make waveform streaming native.

---

## Evaluation snapshot

### Reward design (Expresso, Section 5.2)
- *Similarity* reward prefers the default (undirected) reading in 62.0 %–99.7 % of pairs across styles.
- *Preference* reward with a written direction prefers the **whispered** reading in **62.2 %** (vs. 10.6 % without direction) and the **laughing** reading in **64.4 %** (vs. 27.1 %).

### RL vs. SFT (Seed-TTS-Eval)

| Metric | SFT → RL |
|---|---|
| Flagged first takes (zh) | 2.82 % → **1.24 %** |
| Flagged first takes (en) | 3.40 % → **1.84 %** |
| First-take CER (zh) | 1.66 → **1.00** |
| First-take WER (en) | 3.49 → **2.20** |
| Final CER after regen (zh) | 0.84 → 0.83 |
| Final WER after regen (en) | 1.60 → 1.54 |

### Multi-reference cloning (LibriSpeech-PC test-clean)

| System | WER | SIM | UTMOS |
|---|---|---|---|
| Drama 3 (1 ref) | 1.85 | 0.605 | 4.31 |
| Drama 3 (3 refs) | 1.85 | **0.633** | 4.31 |
| OmniVoice | 1.30 | 0.729 | 4.28 |
| Qwen3-TTS | 1.60 | 0.704 | **4.41** |
| CosyVoice 3 | 1.59 | 0.694 | 4.28 |

Drama 3 is competitive but not SOTA on this benchmark (OmniVoice leads SIM).

### Multilingual cloning (MiniMax Multilingual Testset, 24 languages)

Lowest error among compared systems in Chinese, Dutch, English, German, Turkish. Example CER/WER (%): zh 0.63, en 1.13, de 0.48, nl 0.59, tr 0.52. Weaker spots: Cantonese 22.70 (OmniVoice 17.71), Greek 3.39, Romanian 4.94, Indonesian 4.93.

### Infilling (Ming-Freeform basic, CosyEdit protocol)

| Op | EN WER | EN SIM | ZH WER | ZH SIM |
|---|---|---|---|---|
| Insertion | 3.03 | 0.905 | 2.15 | 0.909 |
| Deletion | 5.79 | 0.838 | 7.82 | 0.827 |
| Substitution | 2.08 | 0.866 | 1.87 | 0.898 |

Beats Cartesia Sonic-3 by a wide margin on both basic and full splits.

### Multi-turn revision (Step-Audio-Edit)

Drama 3 rises across iterations (emotion 55.5 → 58.2 %, style 76.5 → 81.8 %), but Step-Audio-EditX leads emotion and paralinguistic metrics; Drama 3 leads style.

### Inference throughput

| Metric | Drama 3 | S2 |
|---|---|---|
| Single-stream fps (H200) | **271** | 177 |
| Single-stream fps (B300) | **295** | — |
| Weights read per frame | ~**6.0 GB** | ~15.6 GB |

**FP8 experts** raise per-stream rate at concurrency 32 by **36 %** over BF16 on H200. Kernel and scheduling changes add **+1.7 % to +15.3 %** depending on the change. **Prefill–decode disaggregation** on RTX PRO 6000: **~24 streams/GPU** vs. **~10 on a single GPU** at the 35-fps first-percentile bar; unit ready time 47 s vs. 148 s. CER delta between PD and single-GPU serving: **0.02 points**.

Inference is shipped through **SGLang-Omni**.

---

## What's genuinely new vs. recomposition

**New in this report** (worth their own depth files):
- [Codebook-MTP decoder](../multimodal/codebook-mtp-decoder.md) with **partial fusion** (`c_0 + c_1` back into the backbone, scaled by `1/√2`).

**Recomposed from prior work** (strong engineering, not new concepts):
- Hybrid Gated-DeltaNet + full-attention backbone — inherited from Qwen3.5-35B-A3B.
- MoE routing (256 experts, top-8 + shared) — standard DeepSeekMoE-style.
- Dr. GRPO with asymmetric clip, no KL, no reference model — a known composition of GRPO modifications.
- DAC-based 44.1 kHz / 21.5-fps / 10-codebook codec — inherited from S2.
- FP8 expert quantization, prefill–decode disaggregation, SGLang-Omni serving — serving-stack engineering.

---

## Why it matters

- **A real open-source counterpart** to closed studio-TTS stacks (ElevenLabs, Sesame CSM) with the architectural recipe disclosed.
- **An MoE scaling data point for audio** at an LLM-style active fraction (3B / 35B ≈ 8.6 %) — competitive with much smaller dense audio models while using similar active FLOPs.
- **The codebook-MTP decoder with partial fusion** is a design worth reusing elsewhere in residual-codec audio generation.
- **The RL recipe** demonstrates that preference-RM GRPO on a frozen decoder sharpens direction-following and expressivity without regressing acoustic quality.

---

## Sources

- Paper: *Fish Audio Drama 3 Technical Report* — Fish Audio, 2026 — arXiv:2610.18010.
- Codec: Fish Audio S2 (predecessor) — reused DAC-based tokenizer.
- Backbone initialization: Qwen3.5-35B-A3B.
- Serving: SGLang-Omni.
