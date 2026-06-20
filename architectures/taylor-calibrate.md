# Taylor-Calibrate
*Depth — principled initialization for converting a softmax-attention Transformer into a hybrid Gated DeltaNet student.*

**TL;DR:** Linear-attention students (Gated DeltaNet, GLA, etc.) are usually obtained by **distilling** a pretrained Transformer, but naively copying the teacher's attention projections leaves the student's recurrent decay, write gate, and output gate at random — so the first thousands of distillation tokens are spent un-doing a bad starting point. Taylor-Calibrate sets those GDN-specific parameters analytically from a Taylor expansion of the teacher's attention, then does a short per-layer alignment pass. Result: up to 88× better zero-shot students and 5–9× fewer distillation tokens to match recovery targets.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [mla](mla.md)
**Related:** [transformer-block](transformer-block.md)

---

## What it is

Hybrid linear-attention conversion replaces some softmax-attention layers in a pretrained Transformer with Gated DeltaNet (GDN) layers and distills the swapped layers to match the teacher. The teacher's QKV projections can be reused, but GDN has **extra knobs** the teacher never had:

- a recurrent **decay** rate (how fast the state forgets),
- a **write gate** (how much each token modifies state),
- an **output gate** (how much state is read out per step).

Standard practice is to initialize these from heuristics (e.g. exponential decay constants, sigmoid bias = 0) and let SGD sort it out. That's where the brittleness comes from.

## How it works

Two-step initialization, no extra trainable parameters introduced:

1. **Taylor-guided closed-form init.** Expand the teacher's softmax attention to first order around the current token. The first-order coefficients map directly onto GDN's value projection, memory timescale, write gate, and output gate. Read off those parameters from the teacher's attention statistics (typically computed on a calibration set of a few thousand tokens). The converted layer now matches the teacher's output **to first order** at initialization, instead of randomly.

2. **Per-layer alignment pass.** Run a short optimization that minimizes the per-layer output discrepancy between teacher and student for each converted layer in isolation. Costs negligible compute compared to distillation but tightens the per-layer match before joint distillation begins.

Then proceed with standard distillation (logit + intermediate-state matching).

## Why it matters

- Up to **88× improvement** in a representative zero-shot student ablation vs. naive copy-init.
- **4.9× – 9.2× fewer distillation tokens** to reach matched recovery targets, across four teacher settings and three retained-layer policies.
- Linear / hybrid attention is the practical path to long-context inference at reasonable serving cost. Cutting distillation budget by ~5× makes the conversion routine rather than a research project — strategically important as more frontier teams ship hybrid-attention long-context models.

## Gotchas & tricks

- The Taylor expansion is around the *current* token's attention scores; calibration statistics depend on the input distribution. Use in-domain calibration data for the deployment workload.
- The method initializes GDN's recurrent dynamics; it does **not** propose a new GDN architecture — orthogonal to architectural improvements in the linear-attention family.
- "Retained-layer policy" (which layers stay softmax, which become GDN) is itself a hyperparameter. Taylor-Calibrate's gains are reported across three policies, suggesting the init benefit is robust but the optimal layer-mix is still task-dependent.
- The alignment pass is per-layer, so it parallelizes trivially across layers — useful for very deep teachers.

## Sources

- Paper: *Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation* — Zhou et al., U. Sydney / Together AI / Berkeley / UT Austin / Microsoft, 2026 — arXiv 2606.16429.
- Paper: *Gated Delta Networks* — Yang et al., 2024 — the GDN architecture being targeted.
- Related: linear-attention conversion literature (LoLCATs, Hedgehog, MambaInLlama).
