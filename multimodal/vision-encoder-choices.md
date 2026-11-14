# Vision Encoder Choices in Multimodal LLMs
*Depth — pretrained-frozen vs. pretrained-finetuned vs. trained-jointly-from-scratch for the vision encoder in an MLLM.*

**TL;DR:** The usual VLM recipe is to bolt a pretrained vision encoder (CLIP, SigLIP, DINO) onto a pretrained LLM via a small projection. The "Unbundling Multimodal Vision Encoding" 2026 study rigorously ablates alternatives and finds that **an encoder trained jointly from scratch, inside the full MLLM, matches a strong pretrained encoder** — without any separate vision-pretraining stage. Several long-held folk choices (freeze the encoder; use a big projection) fail to replicate under careful control.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/transformer-block.md](../architectures/transformer-block.md)

---

## What it is

A design-space ablation for "how should a multimodal LLM encode images." The study holds everything else fixed (data mix, LLM backbone, projection shape, training compute) and varies only the encoder treatment:

- **Pretrained + frozen.** Classic LLaVA-style: SigLIP/CLIP weights locked, only the LLM and projection update.
- **Pretrained + finetuned.** Encoder weights update during MLLM training.
- **From scratch + joint.** No vision pretraining. Encoder has random init and trains jointly with the LLM from the first step.

For each, it also varies how parameters are adapted (LoRA, full, projection-only) and projection design (single MLP, multi-token Q-former, pixel-shuffle).

## How it works

Experimental design uses an **equal-compute** comparison, which is the piece that distinguishes it from prior work:

- Each variant gets the same total FLOPs (encoder pretraining + MLLM training).
- For from-scratch, all of the FLOPs go to joint MLLM training; the encoder isn't pre-cooked.
- For pretrained variants, FLOPs are split between encoder pretraining (replicated from source) and MLLM finetuning.

The finding: once compute is equalized, the from-scratch joint encoder matches a pretrained encoder on standard MLLM evals. The implicit vision representation that the LLM scaffolds up is sufficient; pretraining buys mostly optimization-ease, not final-quality.

Secondary findings:

- **Freezing the encoder** gives up ~1–2 points of performance for no compute savings at MLLM scale — the "frozen is more stable" folk belief doesn't hold when the data mix is clean.
- **Large projection modules** (long Q-formers) don't outperform a compact two-layer MLP at matched data.
- **Color encoding and image patch ordering** affect quality more than projection design — surprising if your prior was that the projection is the key design choice.

## Why it matters

- If pretrained vision encoders aren't load-bearing, the MLLM design budget shifts. You no longer need a separate vision-pretraining pipeline, which is operationally significant.
- Suggests that "vision pretraining" and "LLM language pretraining" are not independent capability pools — the LLM's text-pretrained representations substantially subsume what a frozen SigLIP contributes.
- Supplies a clean equal-compute baseline the field had been missing. Prior comparisons all varied two things (pretraining amount + architecture) at once.

## Gotchas & tricks

- "Equal-compute" is the whole story — unequal-compute comparisons will still show pretrained encoders ahead, because the pretrain-then-finetune workflow front-loads useful gradient steps.
- From-scratch joint training needs careful early-stage warmup; the encoder's random features lag the LLM's pretrained features for the first few hundred steps.
- Results are specific to data mixes with meaningful text supervision (image captions, OCR). On diagram-heavy or abstract-visual tasks, a vision-only pretrain contributes more.
- Does not generalize to video — video encoders carry temporal priors that from-scratch training at MLLM compute budgets can't reproduce.

## Sources

- Paper: *Unbundling Multimodal Vision Encoding* — Hoang, Qian, Li, Lewis, Ghazvininejad, Zettlemoyer, Beyer, Ren, Verbeek, Xie, Xu, 2026 — https://arxiv.org/abs/2611.17334
