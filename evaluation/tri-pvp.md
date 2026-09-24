# Tri-PvP
*Depth — modality-bias benchmark for omni-modal LLMs that disentangles perceptual and propositional evidence.*

**TL;DR:** Existing modality-bias benchmarks for omni-modal LLMs (vision + audio + text) conflate two distinct forms of evidence within a single modality — **perceptual** (raw photo/recording) and **propositional** (declarative claim about the same thing) — so measured modality bias is confounded with evidence-form bias. Tri-PvP is an 8,000-sample tri-modal conflict benchmark that crosses vision, audio, and text, letting vision and audio each take perceptual or propositional form independently. Five OLLMs show robust visual bias plus a *systematic evidence-form asymmetry*: perceptual-heavy in vision, propositional-heavy in audio.

**Prereqs:** [README.md](./README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [../interpretability/README.md](../interpretability/README.md)

---

## What it is

A conflict-benchmark for omni-modal LLMs: each item presents evidence in vision, audio, and text that disagree, and the model has to answer a question whose correct answer depends on which modality it trusts. What's new is the **factorial design**: for each of vision and audio, the evidence appears as either **perceptual** (raw signal — a photo of a dog, a bark recording) or **propositional** (declarative claim — the text "this is a dog", or an audio recording of the sentence "this is a dog"). Text is always propositional.

Prior benchmarks fixed evidence form implicitly and reported modality bias, without distinguishing "the model weighs vision heavily" from "the model weighs images heavily because they're perceptual, and would also weigh recorded speech heavily if it counted".

## How it works

**Factorial construction.** 8,000 samples crossing:

- Vision evidence form: perceptual (photo) vs propositional (rendered text image).
- Audio evidence form: perceptual (raw recording) vs propositional (recorded speech asserting the claim).
- Text form: always propositional.
- All three modalities disagree; the "correct" answer per axis is what that modality asserts.

**Metrics.**

- Per-modality preference rate: how often the model's answer matches each modality.
- Evidence-form asymmetry: within a modality, preference rate under perceptual vs propositional signals.

**Interpretability probes.** Layer-wise linear probing on hidden representations shows modality bias is linearly decodable from early layers. Contrastive decoding at inference partially — but not fully — mitigates it, indicating the bias is baked into internal representations, not just decoded surface behavior.

## Why it matters

- **Cleanly attributes modality bias.** By separating perceptual and propositional evidence, Tri-PvP lets researchers ask "does this model prefer vision, or does it prefer perceptual signals?" — two different questions.
- **Reveals a surprising asymmetry.** Vision: perceptual-heavy. Audio: propositional-heavy. This inversion is a nontrivial claim about how OLLMs integrate modalities.
- **Bounds surface-level mitigations.** Bias linearly decodable in early layers means prompt-level and contrastive-decoding fixes only patch symptoms.

## Gotchas & tricks

- **Evidence-form definitions are load-bearing.** "Propositional audio" (spoken declarative) and "propositional vision" (rendered text image) are the paper's operationalizations. Alternative forms (captions, subtitles) may give different asymmetries.
- **Five-model sample.** Robust visual bias is claimed across five OLLMs — a small sample. Replicate across new omni releases before generalizing.
- **Contrastive decoding partially mitigates.** Report both raw and mitigated bias; the gap tells you how much surface-level intervention can buy you.
- **Bias vs error is not the same thing.** A model might strongly prefer vision *and* be right about that on natural distributions. Tri-PvP measures preference under *conflict*, which is deliberately adversarial.

## Sources

- Paper: *Tri-PvP: Exposing Modality Bias in Omni-Modal Large Language Models through Perceptual-Propositional Evidence Conflicts* — Piao, Chen, Chu, Chen, Kuan, Lee, Chen (NTU Taiwan), 2026 — [arXiv:2609.06011](https://arxiv.org/abs/2609.06011).
