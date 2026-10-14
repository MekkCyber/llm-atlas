# Active Spatial Mentalization (SpatialASM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Teaches a vision-language model to plan a task-relevant viewpoint and then imagine the scene from it before answering a spatial question. Training-time spatial simulation internalizes this active mentalization so inference needs no external renderer or simulator in the loop. Reports accuracy gains across seven spatial benchmarks and four LVLM backbones.

**Prereqs:** [multimodal/README](../multimodal/README.md)
**Related:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A training-time method, not an architecture. Produces data in which the model (1) proposes a viewpoint target useful for the current spatial question, and (2) is supervised to internalize what the scene *looks like* from that viewpoint. At inference, the model simulates mentally — no simulator is invoked.

The intuition is cognitive: humans doing spatial reasoning actively re-orient. Current VLMs don't, which the paper argues is why they are weak on spatial tasks even when all needed information is in the input image.

## How it works

Three pieces:

1. **Viewpoint proposal.** For each spatial training example, the model is prompted to emit a target viewpoint (position + orientation parameters, or a textual landmark) relevant to answering the question.
2. **Simulator-in-the-loop training.** A spatial simulator renders the scene from the proposed viewpoint. The rendered view (or its latent embedding) is paired with the student's internal "imagined" representation.
3. **Mentalization supervision.** A loss aligns the student's internal viewpoint representation with the simulator's — so after training, the student can produce the same internal representation without the simulator.
4. **Answer head.** Finally the model answers using both the original view and the mentalized viewpoint.

At inference: steps 1 and 3 happen in the model's forward pass; step 2 is dropped.

## Why it matters

- **Fills a known VLM weakness.** Spatial reasoning is a canonical failure mode; this gives a direct training signal rather than relying on scale or data volume.
- **Generalizes across backbones.** Reported gains across four LVLM backbones suggest this is a *training-data pattern*, not an architecture-specific trick.
- **No inference-time rendering.** The simulator is only present during training; deployment cost is unchanged.

Reported: accuracy gains across seven spatial benchmarks and four LVLM backbones.

## Gotchas & tricks

- The simulator is the engineering bottleneck. The paper's results depend on having scenes rich enough to render from arbitrary viewpoints — not all spatial datasets have this.
- "Mentalization" is only as useful as the viewpoint proposal is. Random viewpoints hurt; the proposal must be conditioned on the question.
- Risks hallucination of the mentalized view if the alignment loss is weakened. Reported ablations keep it on throughout training.

## Sources

- Paper: *SpatialASM: Internalizing Active Mentalization for Spatial Reasoning* — Liu et al., Shanghai AI Laboratory, 2026 — [arXiv:2610.13832](https://arxiv.org/abs/2610.13832)
