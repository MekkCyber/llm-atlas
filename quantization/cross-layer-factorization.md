# Cross-Layer Factorization (GeoPair)
*Depth — training-free transformer compression by geometry-preserving cross-layer weight factorization.*

**TL;DR:** Transformers have redundancy across layers, but existing post-training compression pipelines either optimize each layer in isolation or heuristically group layers that "look similar." **GeoPair** finds structurally compatible weight pairs across layers and factorizes each pair through a **shared dictionary**, while still respecting each layer's distinct activation geometry (calibration statistics). Combined with structured sparsity this yields highly efficient decompositions with no training and no fine-tuning, and outperforms independent per-layer decomposition and heuristic pairing baselines across architectures, scales, and modalities.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** *(add to _number-formats table when landed)*

---

## What it is

A post-training compression scheme for transformer weights. Two ingredients:

- **Cross-layer pairing** — find pairs of weight matrices (typically in different layers, same functional role — e.g. two `W_v` matrices) whose column spaces are structurally compatible, so a single shared basis can approximate both well.
- **Geometry-preserving factorization** — the shared dictionary is optimized against each layer's activation covariance, so the reconstruction error is small where each layer *actually operates*, not where the raw weight matrix is largest.

Everything is post-hoc: no gradient updates, no fine-tuning pass, no calibration beyond a small forward-pass activation sample.

## How it works

1. **Collect activation statistics.** Run a small calibration dataset through the model and cache the input-activation covariance for each target layer.
2. **Sequential pairing.** Iterate over layers in order. For each unpaired layer, search over unpaired candidate partners for one whose projection subspace is compatible (measured against the covariances). Score by a joint reconstruction objective under a shared basis.
3. **Optimize the shared factorization.** For each accepted pair, jointly optimize the shared dictionary `D` and per-layer coefficient matrices `A_1, A_2` so that `W_i ≈ D · A_i` minimizes activation-weighted reconstruction error on both layers.
4. **Apply structured sparsity.** On top of the factorization, prune structured blocks of the coefficient matrices where activation-weighted contribution is small.

The result is a lower-rank representation shared across the paired layers plus a small per-layer coefficient tensor.

## Why it matters

Practical training-free compression is one of the few remaining high-leverage inference cost knobs after strong quantization has been applied. Cross-layer factorization has been a heuristic art (SVD each layer, pair by cosine similarity, hope for the best). GeoPair puts pairing and factorization on an optimization footing so the compression ratio is achievable without retraining — critical for closed-weights deployments where any fine-tuning is expensive or forbidden.

## Gotchas & tricks

- The compatibility metric is what makes or breaks the pairing step. Naive Frobenius similarity between raw weights is too weak; activation-weighted alignment is required.
- Sensitive to which functional role you pair — value projections tend to pair across layers well; query projections less so because attention head specialization differs more.
- Complementary to quantization: apply factorization first (reduces parameter count), quantize the resulting factors second. The paper reports the ordering matters and this direction is safer.

## Sources

- Paper: *GeoPair: Geometry-Preserving Cross-Layer Factorization for Training-Free Transformer Compression* — Ali & Lefkimmiatis, 2026 — [arXiv:2609.25963](https://arxiv.org/abs/2609.25963)
