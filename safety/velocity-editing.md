# Velocity Editing for Flow-Matching Safety (VESFlow)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **training-free safety method for few-step flow-matching T2I models** (MeanFlow, rectified flow). Existing safety methods rely on iterative trajectory steering across many denoising steps or on CLIP-style prompt-embedding manipulation — both break under 4-step flow matching. VESFlow instead edits the **velocity field directly** via a Bayesian decomposition of the safe-conditional posterior, while leaving the conditioning prompt untouched. A risk-score gate bypasses editing on benign prompts.

**Prereqs:** [_attacks.md](_attacks.md)
**Related:** [README.md](README.md)

---

## What it is

Flow-matching models learn a velocity field `v(x, t)` that transports noise into samples in a small number of integration steps. Existing diffusion-safety methods exploit the many-step nature of classical diffusion:

- *Iterative trajectory steering* needs many small corrections — can't work in 4 steps.
- *Prompt-embedding manipulation* (e.g., CLIP-centric concept removal) is weakened by modern context-aware text encoders that mix the unsafe concept across the entire embedding.

VESFlow operates on the velocity field — the right primitive for flow matching — instead.

## How it works

The **safe-conditional posterior** for a velocity field is decomposed into a "safe-prior" velocity plus an "unsafe-correction" velocity using Bayes' rule. VESFlow estimates these components from the trained model and the prompt's risk class, then edits the integrated velocity at each of the (few) sampling steps:

1. **Risk score.** A small classifier scores the prompt for unsafe-content risk.
2. **Benign branch.** If risk is low, run the unmodified velocity field — no perturbation, no quality cost.
3. **Unsafe branch (VESFlow):** edit velocity toward the safe-prior component.
4. **Unsafe branch (VESFlow+):** edit *toward* safe and *away from* unsafe simultaneously, by subtracting a scaled unsafe-correction velocity.

The prompt embedding is never modified, preserving prompt fidelity for the benign part of the request.

## Why it matters

- **Closes the safety gap opened by few-step generation.** As the field moves to 1–4 step flow-matching for latency, all the multi-step safety methods have been left behind.
- **Training-free.** No fine-tune, no LoRA, no concept-erasure pass. Drops into an inference stack with the model and a small classifier.
- **Preserves benign quality.** The risk gate ensures benign prompts aren't paying any quality tax.
- **Headline numbers** on MeanFlow / 4-step: NudeNet attack-success drops to 6.3% on Ring-A-Bell and 6.8% on MMA-Diffusion.

## Gotchas & tricks

- **Risk-classifier quality is critical.** A poor classifier either misses unsafe prompts (no editing applied) or trips on benign ones (quality regression). The paper uses a small dedicated classifier; the choice of training data dominates.
- **Velocity decomposition assumes a tractable safe/unsafe split.** For attack types not covered in the training prior (e.g., novel jailbreak phrasings), VESFlow can underperform — the velocity field doesn't "know" the unsafe direction.
- **Step-by-step editing accumulates.** Over 4 steps, repeated small edits compound; over 1 step (extreme MeanFlow), a single edit must do all the work, which can over-shoot quality.

## Sources

- Paper: *Safe Few-Step Generation via Velocity Editing* — Choi, Yoon, NTU Singapore / UNIST, 2026 — [arXiv:2606.23267](https://arxiv.org/abs/2606.23267).
- Background: MeanFlow / rectified flow — the velocity-field generation family VESFlow targets.
- Benchmarks: Ring-A-Bell and MMA-Diffusion — the unsafe-prompt attack suites used.
