# Trajectory-tail steering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Multi-turn diffusion image editing drifts: unedited content slowly degrades across 15–20 turns. TailSteer (2026) is a final-stage training recipe that keeps the early denoising steps on-distribution with the base model and only injects learned "preservation corrections" near the end of the trajectory, supervised by the editor's own rollouts against the source image — no curated edit pairs needed.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** none

---

## What it is

A diffusion edit works by running a denoising trajectory conditioned on an instruction. Chain two edits ("make it sunset", then "add a dog") and the second run inherits the first run's output as its input. Each run slightly moves regions that *should* be preserved (compression artefacts, hue drift). Across many turns the drift compounds and the base image is lost.

TailSteer is a post-hoc finetune of an existing editor model that specifically counters this cumulative drift.

## How it works

The denoising trajectory is split into **head** (noisy end → mid) and **tail** (mid → clean end). Observation: most of the semantic content is already formed by the tail; most of the drift is also introduced in the tail. So:

- **Head:** regularise toward the base model's behaviour (KL / score-matching to the frozen base). Keeps the editor's creative capacity intact.
- **Tail:** learn a *residual correction* that nudges the output toward preserving unchanged regions.

The tail objective is self-supervised. Given a rollout starting from a source image `S`, let the output be `O`. Define a soft "unchanged mask" from the instruction + a vision model, and penalise `||M ⊙ (O − S)||`. This uses only the editor's own rollouts — no human-curated edit pairs.

## Why it matters

Previous approaches to multi-turn edit drift (chain-consistent finetuning, specialised preserve-and-edit encoders) either require expensive paired data or trade off editing quality. TailSteer is cheap, requires only rollouts, and keeps base-model edit quality largely intact. Demonstrated on FLUX, SenseNova, and Qwen image editors over 15–20 turns.

## Gotchas & tricks

- **Mask quality matters.** The "unchanged region" mask drives the self-supervision. A bad mask (either too permissive or too strict) can either fail to prevent drift or lock up legitimate edits.
- **Head/tail split is a hyperparameter.** Papers find a tail of the last 20–40% of denoising steps works across models; moving the split earlier starts to degrade edits.
- **Compatibility with step distillation.** When the editor is already a few-step model, the tail is only 1–2 steps — TailSteer gets less room to correct. Apply before few-step distillation if possible.

## Sources

- Paper: TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing — Li et al., 2026 — [arXiv:2611.06741](https://arxiv.org/abs/2611.06741)
