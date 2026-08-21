# Model lineage verification
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Given two open-weight checkpoints, decide whether they share ancestry — without training data, without activations, from weights alone. Thakur & Khoury (2026) show that the **centered residual signature** of a model — the singular structure of weight residuals after mean-removal per parameter group — is a fingerprint that survives common laundering (permutation, low-rank noise, sign flips, partial merges) and near-perfectly separates related from unrelated public LLaMA-2 checkpoints.

**Prereqs:** *(none — reads only weight tensors)*
**Related:** [README.md](./README.md), [../safety/README.md](../safety/README.md)

---

## What it is

A **data-free, white-box** provenance test. Inputs: two model checkpoints of compatible architecture. Output: a symmetric similarity score in [0, 1] plus a decision threshold. Above the threshold ⇒ shared ancestry; below ⇒ independent.

Not a plagiarism proof — it's a statistical claim that the checkpoints share a training-run history, not that any specific fine-tune happened.

## How it works

1. **Group parameters** into comparable slices (per-layer projections of the same rank/shape).
2. **Center each slice** by subtracting the group-mean tensor. This removes the "typical LLaMA" baseline and isolates fine-tune-specific residuals.
3. **Extract a signature** from the centered residual — the paper uses a spectral summary (singular-value structure + directional statistics) chosen to be invariant under permutation and orthogonal transforms.
4. **Score the pair** with a symmetric distance (e.g. subspace / Frobenius) between signatures. A calibrated threshold decides "related vs unrelated".

Because the pipeline reads only tensors, it needs no dataset, no forward pass, and no gradient — anyone with two `.safetensors` files can run it.

## Why it matters

- **Practical provenance for the open-weight ecosystem.** Enforces licenses, detects undisclosed base models, and audits leaked / laundered checkpoints where naive `sha256` comparison would fail.
- **Robust to standard laundering.** Permutation, low-rank noise, sign flips, and partial merges leave the centered residual's spectral structure largely intact — techniques that fool weight-diff or activation-matching approaches don't fool this one.
- **Complements activation-based fingerprinting.** Activation probes require inference and matched prompts; residual signatures work on frozen artifacts.

## Gotchas & tricks

- **Architecture must match.** Signatures are per-parameter-slice; if one model has been re-architected (added/removed layers, changed head dims), pairing is undefined.
- **Merging degrades the signal.** Two ancestors merged 50/50 sit halfway between the two signatures; strongly-weighted merges are effectively that ancestor.
- **Threshold is calibration-dependent.** The paper's numbers are on the public LLaMA-2 family. Other families need their own calibration — the *ranking* transfers, but the *cutoff* doesn't.
- **Doesn't detect data-level ancestry.** Two teams that trained on the same public corpus from different init will look independent — the fingerprint is of the training run, not the data.
- **Adversarial resistance is future work.** A determined actor who knows about centered residuals could try to whiten the signature; the paper doesn't claim robustness under adaptive attack.

## Sources

- Paper: *Training Leaves Traces: Centered Residual Signatures for Language Model Lineage Verification* — Thakur & Khoury, 2026 — [arXiv 2608.14929](https://arxiv.org/abs/2608.14929) — introduces the signature, the symmetric scoring pipeline, and the LLaMA-2 case study.
