# Flash Primitive Decoding for VLAs (VFlash)
*Depth — emit an entire visual primitive (bbox, trajectory, points) in one forward pass, verified by structural constraints.*

**TL;DR:** VLAs that output structured geometric primitives (bounding boxes, point sets, trajectories) normally emit one coordinate token at a time, autoregressively — slow and error-prone at length. **VFlash** (Wang et al., 2026) decodes the entire primitive in a single parallel forward pass via a lightweight *drafter head*, then validates the draft against structural constraints (ordering, bounds, type) before accepting. Reports lower coordinate error than AR decoding, ~25× faster 3D point-motion forecasting, and 70.2% average success on four MolmoSpaces task suites (vs 62.6% AR baseline).

**Prereqs:** [vla](./vla.md), [speculative-decoding](../inference/speculative-decoding.md)
**Related:** [vla-continual-pretraining](./vla-continual-pretraining.md)

---

## What it is

A **flash primitive** is a tuple of coordinates with a fixed cardinality for a given primitive type:

- 2D bbox: 4 numbers.
- 3D bbox: 6 numbers.
- Point set (fixed N): $3N$ numbers.
- Trajectory (fixed N waypoints, 3D): $3N$ numbers.

VFlash treats the entire tuple as *one* output, predicted jointly from the VLM's conditioning state. In spirit this is [speculative-decoding](../inference/speculative-decoding.md) specialised to geometric structured output: the drafter is trained to directly predict the structure, and a cheap structural verifier plays the role of the target model.

## How it works

### Drafter head

A small head attached to the VLM backbone that, given the VLM's conditioning state, emits all coordinates of a primitive in one forward pass. Trained on annotated primitives with a regression loss; in parallel with the normal autoregressive head so the model retains both decoding modes.

### Structural verifier

A deterministic checker that rejects malformed drafts. Rules depend on the primitive type:

- bbox: $x_1 < x_2$, $y_1 < y_2$, inside image bounds.
- point set: within canonical range, no duplicates beyond a small threshold.
- trajectory: waypoints monotonic along time, segment lengths bounded.

Drafts that fail the verifier trigger a fallback to autoregressive decoding for that primitive.

### Training

Two-head training: at each training step, the AR head is trained normally; the drafter head is trained on the same primitives but regresses the whole structure directly. The structural verifier is used at inference, not training.

## Why it matters

- **Real-time VLA decoding becomes plausible.** Sequential coordinate emission is a well-known bottleneck in VLAs that emit boxes or paths; a draft-and-verify loop specialised to geometric structure breaks it without changing the target distribution (the AR fallback preserves correctness).
- **Transfers beyond trajectories.** The same template — flash-predict a structured output, verify, fall back — applies to any primitive with a fixed shape and a cheap validity test: segmentation masks (checksum valid area), scene graphs (valid edges), tool-call arguments (JSON schema).
- **Composable with other VLA improvements.** VFlash lives at the decoder; it stacks on top of SLA-style continual-pretraining improvements to the backbone.

## Gotchas & tricks

- **Verifier breadth is the limiting factor.** A loose verifier accepts drafts that look structurally valid but are semantically wrong (e.g. a bbox that respects ordering but is in the wrong part of the image). Tighten with task-specific constraints where available.
- **Drafter can collapse on rare primitive shapes.** If a trajectory length is rare in training, the drafter underfits. Fallback to AR decoding handles this correctly but is slower; monitor fallback rate.
- **Head-sharing vs separate head.** A separate drafter head is cleaner but adds parameters; a shared head with primitive-type conditioning is cheaper but needs careful loss balancing.
- **Not a replacement for AR on free-form outputs.** VFlash is primitive-shaped only. Text, variable-length lists, or other unstructured outputs still need AR.
- **Benchmark honesty.** The 25× number is single-rollout preliminary; include end-to-end wall-clock including the verifier in any production evaluation.

## Sources

- Paper: *VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding* — Wang et al., NVIDIA + JHU, 2026 — [arXiv:2611.08449](https://arxiv.org/abs/2611.08449).
- Background: *MolmoSpaces* — the VLA / embodied task suites used in the eval.
- Background: [speculative-decoding](../inference/speculative-decoding.md) — the structural cousin of VFlash's draft-and-verify loop.
