# Cross-Modal Steering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Inside a **unified multimodal model** (one that shares parameters for language and image generation), a steering vector derived from *one* modality can steer generation in the *other* modality **zero-shot**. A text-derived "happy" vector steers an image generation to be happier; an image-derived "small object" vector shifts text toward "small." Transfer is strongest when the two vectors align in direction, despite a clear modality-gap offset between text and image representations.

**Prereqs:** [activation-steering.md](activation-steering.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [activation-steering.md](activation-steering.md)

---

## What it is

Unified multimodal models (shared backbone for both language and image generation; the output head decides modality) make it possible to ask whether *concepts* are shared across modalities inside the model, or kept separate. Zeng et al. (2026) extract a steering vector in one modality — say, "emotion = happy" from text contrasts — and apply it at inference during generation in the other modality (image), with no additional training.

They find transfer in **both directions**, across concepts like emotion and object size. The transferable vectors point in similar directions in the shared residual space, even though the text-generation and image-generation modes occupy clearly offset regions (the "modality gap").

## How it works

### Setup

1. **Target model:** a unified multimodal LLM whose text and image generation share residual-stream parameters.
2. **Source modality:** derive a steering vector in the usual way (see [activation-steering.md](activation-steering.md)) — mean-difference on positive/negative examples in modality $A$.
3. **Target modality:** apply the vector at the same layer during generation in modality $B$. No projection, no fine-tuning.

### The modality gap

Text and image generations use different residual-stream regions on average — a *bias offset* between modalities. But the **direction of concept movement** is cross-modal: adding $v_c$ to an image-mode activation shifts generation along the same conceptual axis as adding it to text-mode.

### Transfer predictor

The paper reports that cross-modal transfer strength correlates with the **cosine similarity** of the vectors derived independently in each modality. When text-derived $v_c^{\text{text}}$ and image-derived $v_c^{\text{image}}$ point in nearly the same direction, transfer is strong. When they diverge, it's weak — modality-specific concept encoding.

## Why it matters

- **Existence proof for cross-modal concept circuits.** Shared internal directions imply the model builds a modality-agnostic concept space, with modality acting as a bias on top.
- **Tooling transfer.** Mech-interp tooling built for text (SAE training, probing, steering endpoints) extends to image generation inside unified models at zero marginal training cost.
- **Debiases a measurement.** The modality gap is primarily a translation, not a rotation. Prior interpretations of multimodal representations should separate the two.
- **Constrains theory.** Any account of multimodal representation in these models has to explain cross-modal direction alignment.

## Gotchas & tricks

- **Only inside *unified* models.** Doesn't apply to pipelined architectures (separate image generator fed by text encoder); the shared backbone is the mechanism.
- **Concept scope.** Transfer works for concepts with clear cross-modal analogues (emotion, size, color). Modality-specific concepts (e.g. "italicized") don't transfer because they have no counterpart.
- **Direction, not magnitude.** Transferable vectors agree in direction; magnitude is modality-specific. Scale $\alpha$ separately when steering.
- **Layer choice carries over.** The layer at which text steering works tends to also work for image steering — further evidence of shared layer-wise semantics.
- **Not a fix for alignment or safety.** The result is descriptive. It makes some safety tooling easier (text-side steering acting on image gen) but doesn't resolve the underlying robustness questions.
- **Watch the baseline.** A randomly oriented vector of similar norm should *not* transfer. Control runs are essential; the paper reports them.

## Sources

- Paper: *Cross-modal Steering Vectors Exist Across Both Image and Text Generation* — Zeng, Subramani, Rott Shaham, Krojer, et al., 2026 — [arXiv:2610.36269](https://arxiv.org/abs/2610.36269)
