# Directional Decomposition of Transformer Updates
*Depth — decomposing each additive residual update into parallel and perpendicular components relative to the current state.*

**TL;DR:** Each transformer layer's additive update to the residual stream can be decomposed into a **parallel** component (aligned with the current state's direction) and a **perpendicular** component. He et al. (2026) show the parallel component is unexpectedly large; that in *value-space*, non-self parallel manipulation is uniquely robust; and that suppressing the aggregate parallel component during pretraining lowers loss and improves downstream averages. A rare mech-interp finding that also delivers a pretraining recipe.

**Prereqs:** [../architectures/transformer-block.md](../architectures/transformer-block.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [README](README.md), [../architectures/reordered-norm.md](../architectures/reordered-norm.md)

---

## What it is

A geometric analysis of how the residual stream evolves through a transformer. For each update `Δ` written by a layer (attention output or MLP output) to a residual state `h`, decompose:

- `Δ_parallel = (Δ · ĥ) ĥ`  — the component along `h`'s current direction (scales, does not rotate)
- `Δ_perp = Δ − Δ_parallel`  — the component orthogonal to `h` (rotates the direction)

The paper applies this decomposition in **two spaces**:

- **Residual space.** Attention and MLP outputs decomposed against the current hidden state.
- **Value space.** Attention's value-aggregation decomposed against the *current token's own value* — so "parallel" here means "keep the self message", "perpendicular" means "add non-self content".

## How it works

Two experimental threads:

**Analysis.** Across pretrained models, He et al. find that the parallel component is substantial (not a small residual-identity leftover). This contradicts a common assumption that layers mostly rotate directions and hardly rescale.

**Targeted edits.** They test edits that modify only the parallel or only the perpendicular component in each space. Key finding: **exclude-self value-space parallel manipulation is markedly more robust** than residual-space or perpendicular edits. Practically, scaling only the non-self aggregate in the value direction preserves reasoning far better than scaling everything.

**Compression-error decomposition.** Applied to compression methods (quantization, pruning): the update error's perpendicular component separates compression methods more cleanly than the parallel error does. Perpendicular error is a better metric than L2 error for comparing compressors.

**Pretraining regularizer.** Adding a term that suppresses the full-aggregate parallel component during from-scratch pretraining lowers validation-loss trajectories and improves downstream benchmark averages, with the value-space variant strongest.

## Why it matters

- **Interpretability.** Provides a language for *how* a layer moves the residual stream, sharper than "attention writes / MLP writes".
- **Model editing.** A safer edit surface — value-space non-self parallel scaling — for interventions like feature amplification or safety-circuit ablation.
- **Compression evaluation.** A better error metric for comparing quantization and pruning methods.
- **Pretraining recipe.** A cheap regularizer that empirically helps.

## Gotchas & tricks

- **Space matters.** Residual-space and value-space parallel edits behave very differently — don't conflate them.
- **Perpendicular ≠ safe.** Perpendicular edits change direction, which downstream layers care about; they are more disruptive than they look.
- **Exclude-self is the trick.** Value-space parallel manipulation that includes the token's own value is destructive; excluding self preserves the direct signal.
- **Regularizer strength is delicate.** Too strong suppression harms model quality; the paper's schedule ramps in over training.
- **Norm-choice matters.** The decomposition is only well-defined with respect to a chosen norm; L2 is default but attention QK norms complicate the picture.

## Sources

- Paper: *Disentangling Representation Evolution in Transformers through Directional Decomposition* — He, Zhang, Yan — 2026 — [arXiv:2609.15975](https://arxiv.org/abs/2609.15975).
