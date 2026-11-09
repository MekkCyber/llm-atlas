# Parallel primitive decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When a VLM has to emit *structured* spatial outputs — points, bounding boxes, trajectories — token-by-token AR decoding is wasteful because the output's schema is already known. Parallel primitive decoding predicts all coordinates of one primitive in a single forward pass using a lightweight drafter whose output layout is parameterized by (primitive-type, dimensionality, length). Introduced for visual planning in VFlash (2611.08449); applicable to any structured output over a fixed grammar.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [multi-head-attention](../architectures/multi-head-attention.md)

---

## What it is

Vision-language models used as world/action models emit visual *primitives* — point, bounding box, trajectory of length $T$. A naive AR head serializes these as a token stream (e.g. `(x1, y1), (x2, y2), …`) and decodes one coordinate token at a time, so emitting a 3D trajectory of length 100 costs 300 decode steps. Parallel primitive decoding replaces that loop with one parallel forward pass per primitive, over a drafter that already knows the primitive's layout.

## How it works

Given a primitive type $p$ (box, point, trajectory, polyline) with schema $(\text{dim}, \text{length})$:

1. The VLM backbone produces a context embedding for "what primitive do I emit next?".
2. A **primitive drafter** $D_p$ — a small decoder with cross-attention to the backbone — outputs an entire primitive's worth of coordinates in one pass. Its output size is $\text{dim} \times \text{length}$; for variable-length trajectories, the drafter also emits a learned EOS head or a length token.
3. **Training objective**: no teacher-forcing against a reference token sequence. The loss is *structural*: regression on coordinates + task-specific geometric correctness (e.g. IoU for boxes, Chamfer/trajectory error for paths). Any structurally valid prediction with low geometric error is accepted — the drafter is free to disagree with the AR teacher's token ordering.

Because training doesn't require exact-token agreement, the drafter can be much smaller than the AR head it replaces, and its latency is dominated by a single matmul.

## Why it matters

- **Latency scales with primitive count, not coordinate count.** A 100-step 3D trajectory costs one forward pass, not 300.
- **Preserves open-ended planning.** The VLM backbone still decides *what* primitive to emit; only the *filling in* of coordinates is parallelized.
- **Decouples from AR token distribution.** The drafter can be trained on annotated primitives alone, without an AR teacher.

VFlash reports ~25× faster single-rollout generation for 3D point-motion forecasting vs AR decoding, and 70.2% vs 62.6% average success on MolmoSpaces when a world-action model consumes VFlash-decoded plans.

## Gotchas & tricks

- **Schema must be known in advance.** Open-ended text output can't use this — only outputs with a closed grammar.
- **Error metric choice drives accuracy.** Pure coordinate MSE under-weights geometric errors (a trajectory's shape matters more than per-point position); tasks with rotation/scale invariance need matched losses.
- **Variable-length primitives** need a separate length-prediction head; otherwise the drafter either pads or truncates.
- The structural-equivalence loss subsumes autoregressive distillation when the AR teacher is itself geometrically correct — so distillation from an existing VLM-as-policy is a free initialization.

## Sources

- Paper: VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding — Wang et al., 2026 — [arXiv:2611.08449](https://arxiv.org/abs/2611.08449)
