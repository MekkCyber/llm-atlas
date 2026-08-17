# Memory Decoder (MemDec)

*Depth — a small, separately-trained memory-augmented decoder that specializes a frozen large backbone to a narrow domain without touching the backbone's weights.*

**TL;DR:** **MemDec** is a small (4B) side-path decoder that adds domain specialization on top of a large *frozen* backbone. Introduced alongside Intern-S2-Preview-397B (2026), it acts as a memory-augmented adapter: rather than fine-tuning the frontier model per scientific domain (expensive, and destroys the shared prior), MemDec is trained per domain while the 397B backbone stays untouched. Reported: `Intern-MemDec-4B` lifts Biology-Instructions from 56.92 → 60.32 with zero backbone updates.

**Prereqs:** [_post-training](_post-training.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../case-studies/intern-s2-preview.md](../case-studies/intern-s2-preview.md), [fine-tuning/README.md](fine-tuning/README.md)

---

## What it is

A separately-parameterized decoder module attached to a large pretrained backbone. The backbone is used **frozen** — no gradient flows into it. MemDec provides a memory-augmented specialization path: its parameters are the ones that update during domain training, and it composes with the backbone's outputs at inference.

Structurally MemDec sits between fine-tuning and retrieval-augmented generation:

- **Full fine-tuning** updates all backbone weights → high domain gains, expensive, destructive to other domains.
- **LoRA / adapter fine-tuning** updates a small parameter delta → cheap, but still lives inside the backbone.
- **RAG** injects domain content at retrieval time → the backbone weights don't move, but there's no *learned* specialization.
- **MemDec** trains a *separate* small decoder as a side-path → learned specialization, no backbone weights modified, no retrieval infrastructure required.

## How it works

The paper describes MemDec as a *memory-augmented decoder* trained per scientific specialization, run alongside a frozen 397B backbone. The composition is designed so the small MemDec captures domain-specific patterns that the backbone can incorporate at inference without any of its own parameters changing.

Detailed architecture and composition mechanism (how MemDec's outputs enter the backbone's forward pass) are not fully spelled out in the released abstract-level material. What is reported:

- MemDec is **4B parameters** in the released `Intern-MemDec-4B` variant (~1% of the 397B backbone).
- It is trained per specialization domain (e.g. Biology-Instructions is one such specialization).
- The backbone is **frozen** during MemDec training — no updates flow to the 397B weights.
- Multiple MemDecs presumably compose (one per domain), though composition rules across simultaneous specializations aren't detailed in the accessed material.

## Why it matters

- **Non-destructive specialization.** Frontier-scale models are expensive to fine-tune per domain, and doing so degrades their general capabilities. MemDec avoids both costs.
- **Modular specialization portfolio.** Every domain gets its own MemDec — swap-in / swap-out at deployment, no interference between domains.
- **Cheap to iterate.** A 4B module trains in a fraction of the time and compute of a full-model fine-tune.
- **Empirical evidence.** `Intern-MemDec-4B` on Biology-Instructions: **56.92 → 60.32** with the 397B backbone unchanged.

## Gotchas & tricks

- **Ceiling is the frozen backbone.** MemDec cannot exceed what the backbone can express when properly conditioned; if the backbone lacks the primitive capability, no MemDec fixes it.
- **Composition across MemDecs.** Loading multiple MemDecs for multi-domain inference is under-specified in the released material — implementing this cleanly likely requires further engineering (routing, additive vs. gated composition).
- **Distinct from LoRA-family adapters.** LoRA/QLoRA parameters *live inside* the backbone as low-rank deltas; MemDec is a *separate decoder*. The compute path is different and the training/serving abstractions are correspondingly different.
- **Underspecified attention/composition mechanism.** The paper describes MemDec at the system level rather than the operator level; verify against the released code/paper before implementing.

## Sources

- Paper: *Intern-S2-Preview: Scientific Agentic Foundation Model* — Shanghai AI Lab (Bai, Chen, Lin, Guo, Zhou et al.), 2026, [arXiv:2608.13505](https://arxiv.org/abs/2608.13505) — introduces MemDec alongside the 397B backbone.
