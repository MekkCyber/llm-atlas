# Multimodal Embedding Training Curriculum
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A three-stage recipe for turning a decoder-pretrained backbone into a unified multimodal embedder: **(1) text pre-finetuning with distillation**, **(2) multimodal pre-finetuning with distillation only across all modality pairs**, **(3) multimodal finetuning with contrastive + spread-out + Matryoshka losses on higher-quality data**. Introduced in EmbeddingGemma 2 (DeepMind, 2026), where ablations show that each stage transition — in particular the Stage 3 switch from distillation to contrastive — is responsible for a measurable chunk of final performance. Followed by checkpoint merging and tail-patch QAT.

**Prereqs:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [../case-studies/embedding-gemma-2.md](../case-studies/embedding-gemma-2.md)

---

## What it is

Training a *multimodal* embedding model is not just text-embedding training with extra modalities concatenated. The modalities interfere during training when the shared space is still forming, and the right objective at each stage is different from the next. The curriculum formalizes the stage structure:

```
decoder-pretrained backbone (causal LM)
          │
          ├── convert attention: causal → bidirectional
          ├── remove per-layer embedding lookup (if any; EG2 does this)
          └── add pool + proj + L2 norm head
          │
     ┌────┴────── Stage 1: Text pre-finetuning  ────┐
     │             ~300B tokens, mostly distillation │
     │             + small contrastive               │
     └────┬───────────────────────────────────────────┘
          │
     ┌────┴────── Stage 2: Multimodal pre-finetuning ─────┐
     │             ~2T tokens, distillation only           │
     │             all modality pairs + mixed, hard negs   │
     └────┬───────────────────────────────────────────────┘
          │
     ┌────┴────── Stage 3: Multimodal finetuning ────────┐
     │             ~600B tokens, contrastive +            │
     │             spread-out + MRL                        │
     │             higher-quality task-specific data       │
     └────┬───────────────────────────────────────────────┘
          │
          ├── Stage 4: Checkpoint merging (unweighted avg along trajectory)
          └── Stage 5: Tail-patch QAT, distilled from float checkpoint
```

## How it works

### Why distillation first

Early training sees activations in strongly modality-specific regions (the "modality gap"). Contrastive losses, which treat all pairs symmetrically, tend to collapse or oscillate in that regime. A distillation loss against a strong teacher (another frozen embedding model) gives a dense per-sample target that works even when the shared space is still disorganized. EG2 uses Gemini Embedding 2 as the teacher and distills over **truncated** teacher embeddings so Matryoshka truncation targets are learned alongside the main objective.

### Why contrastive last

By Stage 3, the shared space is organized enough that contrastive losses can refine it. The paper combines **contrastive NCE** (in-batch, hard, same-tower negatives), a **listwise Plackett–Luce** extension, a **spread-out** (second-moment) regularizer, and **Matryoshka Representation Learning (MRL)** across truncation dims (768/512/256/128). Switching to contrastive earlier hurts (ablation: 52.28 overall vs. 55.46 with the curriculum).

### Why the merge and QAT at the end

Checkpoint merging unweighted-averages points along one finetuning trajectory, reducing end-of-run variance. Tail-patch QAT is a short QAT stage distilled from the **frozen float merged checkpoint** — backbone to int4, vision encoder to W8A8. Essentially free: 61.36 → 60.84 on MTEB Multilingual Mean(Task).

### Data mix per stage

| Stage | Volume | Data | Loss |
| --- | --- | --- | --- |
| 1 — Text pre-ft | ~300B tok | QA, sentence sim, code retrieval, web search; multilingual | Distillation (mostly) + small contrastive |
| 2 — Multimodal pre-ft | ~2T tok | Image/audio/video/doc, unimodal + cross-modal + mixed; hard negatives | Distillation only |
| 3 — Multimodal ft | ~600B tok | Task-specific: tool retrieval, audio captioning, medical imaging, etc. | Contrastive NCE + spread-out + MRL |

## Why it matters

- **Beats one-stage multimodal training by ~3 pt** on MMEB-v2 overall (52.28 → 55.46 in the paper's ablation). Non-trivial for a late-stage design choice.
- **Portable recipe.** Nothing in the stage structure is EG2-specific. Any decoder-pretrained backbone + a strong teacher can be converted to a multimodal embedder with this template.
- **Interacts with deployment knobs.** The MRL loss during Stage 3 is what makes truncatable dimensions work at inference; the QAT tail is what makes int4 serving nearly free.

## Gotchas & tricks

- **Teacher quality caps everything.** Stage 1–2 distillation inherits the teacher's biases and ceiling.
- **Vision encoder trainable, audio frozen.** EG2 freezes USM but lets vision train — freezing vision hurts (52.27 → 48.18). Modality-specific calls, not a universal rule.
- **Spread-out loss weight is small but matters.** Removing it collapses the embedding norm distribution over long training.
- **Stage 3 LR is *lower* than Stages 1–2.** Opposite of typical "finetune then LR decay." Preserves the shared space built in Stage 2.

## Sources

- Paper: *EmbeddingGemma 2: Unified and Efficient Multimodal Representations* — Schechter Vera, Dua, Andonov, Smoot, et al. (Google DeepMind), 2026 — [arXiv:2610.37364](https://arxiv.org/abs/2610.37364)
- Related: *Matryoshka Representation Learning* — Kusupati et al., 2022 — the truncation targets used inside the Stage 3 MRL loss.
