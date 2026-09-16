# Streaming Multimodal Reasoning
*Depth — a reasoning-time control protocol for omni-modal (video + audio + text) models that must produce answers as evidence arrives, without letting one modality prematurely commit the model to a claim.*

**TL;DR:** Streaming omni-modal models can *see* something before they *hear* the corresponding audio, and vice versa. If a visually-supported claim is written into memory as a fact before the audio has arrived to confirm or refute it, later reasoning keeps relaying the claim even when the audio contradicts it. **Omni-Streaming Thinking (OST, 2026)** fixes this with **pending claims** linked to modality-scoped verification intervals: each claim is refutable at the end of its verification interval, and refutation triggers a state update rather than a silent overwrite. Reduces vision-induced auditory hallucinations by a large margin.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A **streaming omni-modal model** consumes an unbounded video-plus-audio stream and must decide *what* to answer and *when* to answer from the evidence observed so far. The naive control policy — commit to a claim as soon as any modality supports it — has a specific failure mode: **premature cross-modal commitment**. A frame of video can support a claim before the audio track that would refute it arrives; if the claim then enters the model's reasoning state as a fact, downstream chain-of-thought carries the wrong claim forward.

OST turns claims into first-class objects with three properties:

1. **A pending status** — the claim exists but is not yet a fact.
2. **A verification interval** — a time window during which contradictory evidence *from a specified modality* would refute it.
3. **A modality register** — audio and visual evidence are stored separately, not fused into a single "memory".

At the end of each pending claim's verification interval, OST checks the claim against the evidence in the specified modality. On contradiction, a refutation process reduces the influence of the claim and its dependents, and guides a state update using the new evidence. An **answer gate** decides whether the claims relevant to the user's answer have met the conditions for a response.

## How it works

The mechanism as described in the OST paper:

- **Structured generation.** The model emits *structured outputs* rather than free prose: evidence observed so far, forecasts of future evidence, and claims grounded in that evidence. Each claim is tagged pending and linked to a future verification interval.
- **Modality-scoped evidence stores.** Audio and visual evidence are separate registers; a claim's verification interval names which modality can refute it.
- **End-of-interval check.** At the interval boundary, OST queries the specified modality's register and detects contradictions between the claim and the accumulated evidence.
- **Refutation and state update.** A contradicted claim's influence is reduced, its dependents are recomputed, and the new evidence guides a state update. The state is not merely overwritten; the claim's downstream consequences are recomputed.
- **Answer gate.** An explicit gate decides whether to answer now, given which answer-critical claims are still pending. This decouples "have enough evidence" from "have said enough."

Reported implementation: a frozen Qwen3-Omni-30B-A3B-Instruct backbone with lightweight adaptation (no full fine-tune), plus the structured-output protocol above.

## Why it matters

- **Names a specific streaming failure.** Premature cross-modal commitment is a concrete, reproducible mode; hardening against it is a genuine capability improvement, not a metric tweak.
- **Structural, not statistical.** Pending claims and verification intervals are a structural fix (like [long-CoT RL](../post-training/reasoning/long-cot-rl.md)'s explicit reflection tokens) rather than more data or more scale.
- **Reduces vision-induced auditory hallucinations.** OST reaches d' = 2.95 on OST-DiagBench (fixed video, edited audio) vs ≤ 1.38 for open baselines — a large gap on a diagnostic benchmark that isolates the failure mode.
- **+10% relative average on five streaming/audio-visual benchmarks** over the strongest open baselines, backing up the diagnostic result on standard evaluations.

## Gotchas & tricks

- **Verification intervals need to be right.** Too short and everything is unverified; too long and the model sits on stale pending claims. The paper's ablations on interval length are the ones to read.
- **Separate modality registers cost memory but win reliability.** Fused memory is cheaper but reintroduces the commitment failure this whole method exists to prevent.
- **Answer gate can be conservative.** If the gate is tuned strictly, streaming responsiveness suffers. The paper's tuning is toward reliability; deployment may need a more responsive setting.
- **Frozen-backbone adaptation is fragile at scale.** Lightweight adaptation to a frozen omni-model is a strong plus for reproducibility but may leave headroom on the table compared with full-fine-tune baselines.

## Sources

- Paper: *Omni-Streaming Thinking* — Du, Liu, Zheng, Li, Guo, Zhang, Zou — 2026 — [arXiv:2609.15128](https://arxiv.org/abs/2609.15128).
- Related benchmark: *OST-DiagBench* (introduced in the same paper) — a diagnostic benchmark that fixes video and edits audio to test agreement, absence, contradiction, coexistence, and subtitle-speech conflict.
