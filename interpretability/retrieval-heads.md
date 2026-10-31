# Retrieval Heads
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A *sparse* set of attention heads in a transformer that are specifically responsible for moving evidence from the context into the generated output. Ablating them removes the model's ability to retrieve from long context, including for needle-in-haystack style tasks. The 2026 "Broader Look" paper extends the concept to multimodal LLMs: paired "spoken-or-shown" probes with identical answer content show that retrieval heads stay sparse in audio-visual input and are *mostly shared across modalities* — fewer than 2% of heads carry the cross-modal transfer. Training only these heads with a ranking objective raises Qwen3-VL-8B page Recall@1 on MMDocIR by +6.03 points.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** none

---

## What it is

Not every attention head does the same job. **Retrieval heads** are the (small) subset of heads whose attention weights move information *from a specific context token to the output position* in a way that is causally necessary for the output. They were first identified in text LLMs (Wu et al. 2024) by: for a known evidence token at position $k$, measure which heads attend to position $k$ *and* which heads' removal breaks the output.

A retrieval head is distinguished from a "look-at" head: the latter attends somewhere, but ablation does not change the output. Retrieval heads are those where *attention + causal necessity* coincide.

The 2026 paper's additions:

- **Cross-modal generalization.** Build paired inputs where the same answer is either in audio ("spoken") or in a frame ("shown") — content identical, modality differs. Retrieval heads stay sparse in the audio-visual regime and most of them are *shared* with the text-only retrieval heads of the backbone.
- **Omni-model persistence.** Across 18 LM / VLM / omni-modal checkpoints, omni-modal fine-tunes *largely preserve* the retrieval heads of their language-model backbone.
- **The shared cross-modal subspace is <2% of heads.** But ablating them also damages document understanding, spatial perception, and visual math — the same subspace serves multiple tasks.

## How it works

- **Identification.** Needle-style evaluation with a known evidence position. Record which heads attend to it. Ablate each candidate; keep heads whose ablation drops output accuracy on the needle task.
- **Multimodal probes.** Paired (spoken, shown) inputs with matched evidence content let you compare retrieval heads across modalities on *the same answer*.
- **Training target.** The paper trains only the retrieval heads' attention with a ranking objective (score the correct evidence passage above distractors). This is a very narrow fine-tune — <2% of all heads — that still transfers.

## Why it matters

Three distinct uses:

1. **Diagnosis.** The sparse-and-necessary structure explains why "long-context" models can fail unpredictably: a few heads carrying the mechanism are vulnerable to anything that disrupts them (quantization, aggressive pruning, SFT drift).
2. **Low-cost intervention.** You can improve retrieval by training only the retrieval heads. +6.03 pt Recall@1 on MMDocIR for Qwen3-VL-8B is a lot of signal from <2% of parameters.
3. **Cross-modality mech-interp.** Shows the *same* heads serve retrieval across text, image, and audio evidence — stronger evidence than previously that the "retrieval head" concept names a real functional subspace, not an artifact of one task.

## Gotchas & tricks

- **Identification is probe-dependent.** The specific needle probe choices (position, distractor density, evidence form) affect which heads qualify. Robust definitions use multiple probes and take the intersection.
- **Preserved during omni fine-tune, but not guaranteed.** Across 18 checkpoints the backbone's retrieval heads *largely* persist through omni-modal fine-tuning. "Largely" leaves room for drift; worth re-identifying after post-training stages that reshape attention.
- **Narrow fine-tune, broad side effects.** Training only retrieval heads changes a shared subspace, so other tasks that depend on it (document VQA, spatial reasoning) also move — not just retrieval metrics. Could help, could hurt; worth measuring.
- **Not the same as "attention lens" induction heads.** Induction heads implement $A, B \ldots A \to B$ pattern-completion; retrieval heads are an attention-at-a-known-position mechanism. Partial overlap, not identical.

## Sources

- Paper: *A Broader Look into MLLM's Retrieval Heads* — J. Wang, Y. Li, J. Li, Z. Lin, M. Yu, Institute of Information Engineering (CAS) · Tencent, 2026 — arXiv:2610.32411 — the multimodal and training-target extensions.
- Paper: *Retrieval Head Mechanistically Explains Long-Context Factuality* — Wu et al., 2024 — original retrieval-head identification in text LMs.
