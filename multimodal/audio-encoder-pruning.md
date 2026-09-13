# Audio-Encoder Depth Pruning for Speech LLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In speech-LLM stacks, the audio encoder is disproportionately expensive but naïvely removing encoder blocks causes the decoder to emit garbage — deletion errors, premature EOS — because it is exquisitely calibrated to the original encoder's embedding statistics. **X-AuT** (2026) prunes progressively via short behavioral probes, then restores accuracy with four repair steps: **representation alignment**, **cross-scale distillation**, **scheduled student-policy supervision**, and **LoRA finetuning**.

**Prereqs:** [README.md](./README.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** none

---

## What it is

Depth pruning removes entire transformer blocks from a trained model. It works well on decoder-only LLMs because subsequent blocks re-adapt through residual connections. It breaks badly on encoder-decoder speech models: the decoder was trained against a specific encoder-output distribution, and dropping an encoder block perturbs that distribution in ways the decoder cannot silently absorb — so it starts skipping tokens and terminating early.

X-AuT is a **progressive** pruning framework: never remove multiple blocks at once, and use a four-part repair recipe after each removal.

## How it works

Loop over pruning candidates:

1. **Behavioral probe** — a short evaluation set that flags which layer combinations, when dropped, cause the fewest downstream errors on ASR-like tasks. Cheap first pass to rank candidates.
2. **Remove the top candidate.** Now the encoder has $L{-}1$ blocks.
3. **Representation alignment** — train the student encoder's remaining blocks to produce features that match the teacher encoder's outputs at the decoder's interface layer. Standard feature-distillation.
4. **Cross-scale distillation** — several sibling models at different pruning depths all distill into one student, sharing gradient signal so the student's representation gracefully degrades across depths rather than cliff-drops at one depth.
5. **Scheduled student-policy supervision** — start with teacher-forcing (decoder sees ground-truth tokens), gradually shift to letting the student generate its own tokens under supervision. Avoids exposure bias that would otherwise appear after pruning.
6. **LoRA fine-tuning** — final adaptation of a small adapter on the decoder to soak up the residual distribution shift without disturbing the base decoder weights.

Iterate for the next pruning removal.

## Why it matters

- **Speech-LLM serving is encoder-bound.** In common deployments the audio encoder can be as expensive as the decoder itself; depth pruning is one of the few tools that reduces its cost proportionally.
- **Composable with normal LLM tricks.** Because the decoder is unchanged (only a small LoRA layer added), standard LLM optimizations — KV cache, quantization, speculative decoding — still apply on the language side.
- **Progressive pattern generalizes.** The "one block at a time + multi-step repair" approach is applicable to any encoder-decoder stack (VLM projectors, ASR pipelines, multimodal caption generators) where naïve pruning collapses.

## Gotchas & tricks

- **Never remove multiple blocks between repairs.** The probe scoring may say block A and block B are both safe individually, but together they compound: the decoder cannot absorb double shift in one step.
- **Cross-scale distillation is critical.** Without it, students at different pruning depths diverge; with it, they share a smooth "descent path" through model sizes.
- **LoRA rank matters.** Too low and the decoder cannot absorb the residual shift; too high and you eat the pruning savings. The paper reports a small rank suffices.
- **EOS calibration is the failure mode to watch.** Speech LLMs terminate via a learned EOS boundary; a shifted encoder distribution most often manifests as *premature* EOS, not gibberish. Monitor length distributions during pruning to catch this early.
- **Not applicable to width pruning.** X-AuT is a *depth* pruning framework; width pruning (removing channels) has a different failure surface and different repair needs.

## Sources

- Paper: *X-AuT: Progressive Audio-Encoder Compression for Speech LLMs with Cross-Scale Distillation* — Zhang, Zou, Chen, Yu, Fan, Ding, Li, Zhou, Liu, Huang (XPENG AI), 2026 — arXiv:2609.11412.
