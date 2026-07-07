# Attention Attribution
*Depth — training-free attribution from a VLM's prefill-pass attention to source evidence in a document.*

**TL;DR:** Grounded QA systems need to point each part of their answer back to a specific span of source evidence. Standard approaches either ask the model to cite in-line (prompt-based) or train a dedicated attribution model. Both are expensive and hallucinate. Attention attribution reads the model's *own* prefill-pass attention on a small hand-picked set of heads: those heads consistently spike on source spans that support the answer. Threshold the head-level activations against a calibrated baseline and you get a per-token attribution map in one forward pass — no prompting, no training, no separate model. MultAttnAttrib (Adobe et al., 2026) demonstrates the recipe for multimodal (text + image) long-document QA, matching frontier prompting-based attribution at ~1/7 the latency.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../multimodal/README.md](../multimodal/README.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

For grounded QA the failure mode isn't wrong answers — it's *plausible answers with fabricated sources*. Deployed AI assistants ship attribution as a first-class output; users see the sources and trust the answer only if the sources support it. Attribution methods break into three families:

- **Prompt-based.** Ask the model to emit `(answer, cited spans)` in a structured format. Cheap, but the model can and does fabricate the citations.
- **Trained attribution models.** A separate model learns to score span-answer alignment. Accurate but expensive to train, per-domain.
- **Attention-based (this).** Read the model's already-computed attention as the attribution signal. No new model, no prompt round-trip.

The attention-based family sits on a body of interp evidence that specific transformer heads carry specific roles: copy heads, induction heads, name-mover heads. In the QA setting, some heads reliably spike on source spans that ground the model's output. Attention attribution is the applied side of that observation.

## How it works

**Setup (one-time per model).**

1. **Head selection.** On a small labeled dataset (source spans + verified attributions), sweep every attention head. For each, measure the correlation between its per-token activation on prefill and ground-truth attribution labels. Keep the top-scoring heads — typically 5–20 out of hundreds.
2. **Threshold calibration.** For each selected head, calibrate a per-head threshold that trades precision and recall on the labeled set. Store the head set and thresholds as an attribution "profile" per model.

**Attribution (per query, one prefill pass).**

1. Run the VLM's prefill pass over `(question, document)`. Multimodal: document mixes text tokens and image tokens.
2. Extract the per-token attention activations on the selected heads.
3. Apply the calibrated thresholds. Tokens where enough selected heads exceed threshold become attribution candidates.
4. Aggregate contiguous candidates into spans. For images, the spans become bounding regions on patches.
5. Return per-answer-component spans.

No answer-generation call is needed; the attribution is available at prefill time. This is what gives the ~7× latency win over prompting: prompting requires an extra generation pass through the base model to emit the citations.

## Why it matters

- **Deployable behind existing VLM endpoints.** No fine-tuning, no separate attribution model, no prompt schema. Just add a small extraction pass on the prefill outputs.
- **Multimodal by construction.** Same recipe treats text tokens and image patch tokens uniformly; a head that grounds text answers on text spans also grounds image-referring answers on image regions.
- **Grounds an interp observation in production usefulness.** "Specific heads carry specific roles" is well-established mechanistic interp folklore. Attention attribution is the practical demonstration that a small hand-picked head set is enough for a real user-facing feature.
- **Latency win at scale.** ~7× lower inference latency than prompting-based direct attribution on the same base model is the difference between a UX-viable feature and a batch-only pipeline.

## Gotchas & tricks

- **Head selection is model-specific.** The set of "attribution heads" for Llama-3-VL is not the set for Qwen2-VL. Re-calibrate for each base model.
- **Fine-tuning shifts the head set.** Any post-training that touches attention (instruction tuning, RLHF for that model) can shuffle which heads carry the signal. Re-calibrate after post-training changes.
- **Threshold calibration is safer per-head than global.** Different heads have different natural activation scales; one universal threshold under-reports on quiet heads and over-reports on loud ones.
- **False positives cluster.** Attention heads tend to attend to *adjacent* tokens; naive thresholding produces short, dense spans. A minimum-span filter or contiguity-based smoothing improves precision.
- **Doesn't generalize outside the training modalities.** A head profile calibrated on text + image documents may not transfer to text + video without recalibration on video-token positions.
- **Non-negotiable at long context.** At 100k+ context, all attention head activations get small and noisy; a percentile-based rather than absolute threshold works better than fixed calibration.

## Sources

- Paper: *MultAttnAttrib: Training-Free Multimodal Attribution in Long Document Question Answering* — Tran et al., Adobe Research (+ collaborators), 2026 — [arXiv:2607.01420](https://arxiv.org/abs/2607.01420)
- Related interp: *In-context Learning and Induction Heads* (Olsson et al., 2022), *Interpretability In The Wild* (Wang et al., 2022) — foundational head-role literature this recipe rests on.
