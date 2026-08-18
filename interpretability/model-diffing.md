# Model Diffing (SAE-Feature Diff)
*Depth — compare two models' sparse-autoencoder features to isolate what a fine-tune or adaptation actually changed.*

**TL;DR:** Train sparse autoencoders (SAEs) on two related models — a base and an adapted variant — and diff the resulting feature dictionaries to isolate the features that owe their existence to the adaptation. MMDiff instantiates this for base-LM vs multimodal-adapted MLLMs, producing feature-level interfaces for detection, causal ablation, and steering. Removing discovered features degrades targeted behaviors (spatial −12%, OCR −17%, multimodal-attack success −24%) *with no VQA hit*.

**Prereqs:** none *(SAE background helpful — see interpretability README).*
**Related:** [../safety/_attacks.md](../safety/_attacks.md)

---

## What it is

An interpretability technique for *pairs* of models. If model B is model A plus additional training (multimodal adaptation, safety fine-tune, RLHF, domain SFT), a feature-level diff between SAE dictionaries trained on A and on B pinpoints which internal features are net-new or reshaped by that training. The diff is causal: you can ablate diff-features in B and see behavior revert toward A.

## How it works

For a pair `(A, B)` at the same hidden layer:

1. **Collect activations.** Sample `N` inputs and record hidden states from A and from B at the target layer.
2. **Train two SAEs.** `SAE_A` on A's activations, `SAE_B` on B's activations, same width and sparsity target. (Alternative: train one SAE jointly with a domain flag.)
3. **Align features.** Match features across dictionaries by cosine similarity of decoder directions or by co-activation on shared inputs.
4. **Diff.** Classify features as: (a) shared (present in both), (b) B-only (created by adaptation), (c) A-only (suppressed by adaptation).
5. **Causal validation.** For B-only features suspected of driving a behavior, ablate their directions from B's residual stream at inference and measure behavior change.
6. **Contrastive detection.** Per-token firing analysis on paired prompts (e.g., safe vs unsafe multimodal input) isolates which B-only features fire only on the target class.
7. **Steering.** Add/subtract discovered feature directions to steer B's behavior at inference.

## Why it matters

- **Attribution.** Turns "the fine-tune changed behavior X" into "these specific features drive behavior X." That's what auditing and post-fine-tune red-teaming need.
- **Targeted defense without capability loss.** MMDiff removes multimodal-attack success by 24% with no VQA regression — a clean surgical intervention that broad activation-steering baselines can't match.
- **Generalizes.** The recipe is model-family-agnostic (MMDiff shows it on LLaVA-MORE, PaliGemma 2, InternVL3.5); nothing in it depends on the multimodal setting specifically.

## Gotchas & tricks

- Feature alignment across two independently-trained SAEs is the hardest step; jointly-trained SAEs with a domain flag avoid this.
- Diff features can drift across layers; pick the layer where the adaptation-induced signal is strongest by measuring KL(B‖A) across layers first.
- "Removed" features often reappear compensatorily at nearby layers — always validate causal removal by end-to-end behavior change, not just activation zeroing.
- SAE reconstruction error is the ceiling on what the diff can see; use large-width SAEs at high sparsity for adaptation studies.

## Sources

- Multimodal Model Diffing for Feature Discovery and Control (MMDiff) — Hunar Batra et al., 2026 — [arXiv:2608.09928](https://arxiv.org/abs/2608.09928)
