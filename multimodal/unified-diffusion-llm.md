# Unified Diffusion LLM
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A single **discrete diffusion** backbone handles both visual understanding and image generation inside one model. Shares all parameters across modalities; outputs are iteratively refined over multiple diffusion steps instead of generated autoregressively or sampled from a continuous flow. The leading recipe (Nemotron-Labs-Diffusion-Uni, 2026) combines **cross-paradigm on-policy distillation** from AR vision-language models and continuous flow models with an **elastic self-refining diffusion process**.

**Prereqs:** [cross-paradigm-distillation.md](cross-paradigm-distillation.md)
**Related:** [cross-paradigm-distillation.md](cross-paradigm-distillation.md) · [../inference/spot-speculative-decoding.md](../inference/spot-speculative-decoding.md)

---

## What it is

A unified multimodal model that is **not** a chimera of an AR text decoder and a continuous flow image head. Instead, one discrete diffusion model emits both text and image tokens, trained jointly. Understanding = diffuse over text tokens conditioned on image + prompt. Generation = diffuse over image tokens conditioned on text.

The appeal is uniformity: one objective, one inference kernel, one serving stack. The challenge is matching the quality of specialist stacks that were trained for one modality.

## How it works

Two main contributions in NLD-Uni (Li et al., 2026):

### 1. Cross-paradigm on-policy distillation (OPD)

Teachers:
- An AR vision-language model (strong understanding, weak generation).
- A continuous flow image model (strong generation, no understanding).

Student: the discrete diffusion LLM.

The distillation uses on-policy student samples — rather than pre-generated teacher samples — scored against both teachers' preferences. See [cross-paradigm-distillation.md](cross-paradigm-distillation.md) for the mechanism.

### 2. Elastic self-refining diffusion process

Standard discrete diffusion masks tokens hard: a token is either `[MASK]` or committed. The elastic process **gradually corrupts** tokens over training (soft noise schedule in token space), teaching the model to work with partially committed states.

At inference, the model emits intermediate tokens *before* the diffusion is complete, then iteratively refines them — "commit then refine" rather than "commit at the end." This gives:
- Early visible output (lower latency to first token for text).
- The ability to re-decide, unlike pure AR.
- A natural self-correction loop.

## Why it matters

- **State-of-the-art among unified diffusion LLMs** on VQA, multimodal reasoning, object grounding, text-to-image, and image editing.
- **One serving stack for understanding + generation.** No separate pipelines, no modality routing at the system level.
- **Alternative future for multimodal.** AR-text + flow-image is the current production shape; unified diffusion is a bet that one paradigm, done well, is cleaner.
- **Discrete tokens, not continuous latents.** Keeps the model in token space throughout — easier to combine with tokenizer tricks and speculative decoding (see [../inference/spot-speculative-decoding.md](../inference/spot-speculative-decoding.md)).

## Gotchas & tricks

- **Inference compute.** Multiple diffusion passes are not free; the model's throughput per generated token is below AR under naive settings. SPOT-style speculative decoding closes much of the gap.
- **Teacher availability.** OPD needs *both* an AR VLM and a flow model as teachers. If one is weak, the student inherits its weakness in that modality.
- **Elastic schedule tuning.** The corruption schedule interacts with the number of inference steps — too aggressive, and intermediate commits are garbage; too gentle, and training is slow.
- **Discrete-only ceiling.** Discrete tokens cap the detail achievable in generation vs. continuous flow at the same budget. Modern image tokenizers close most of the gap, but not all.
- **Comparison target.** Compare against *unified* baselines (prior unified diffusion LLMs, Chameleon-style AR unified models), not specialist models of similar parameter count — the right question is "does paradigm unification cost quality?"

## Sources

- Paper: *Nemotron-Labs-Diffusion-Uni: Elastic Self-Refining Unified Diffusion LLM with Cross-Paradigm On-Policy Distillation* — Li, Ye, Heinrich, Fu, Zhang, Liu, Grover, Molchanov, 2026 — [arXiv:2610.36466](https://arxiv.org/abs/2610.36466)
