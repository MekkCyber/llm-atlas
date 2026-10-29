# Case Study: EmbeddingGemma 2

*An open, ~744M-parameter unified embedding model from Google DeepMind that represents text, images, audio, video, and mixed inputs in a single 768-dim space. Interesting not for the final numbers but for the shape of the training recipe (multi-stage with explicit modality + objective progression), the architectural simplification of per-layer embeddings, and a deployment story built around run-time knobs (selective encoder loading, token budgets, truncatable dims, int4 QAT).*

**Related concepts:** [multimodal-embedding-curriculum](../multimodal/multimodal-embedding-curriculum.md) · [fp8](../quantization/fp8.md) · [_data-curation](../data/_data-curation.md)

---

## What this is

EmbeddingGemma 2, released October 2026 by Google DeepMind. A single ~744M-parameter model that produces one unified embedding for text, images, audio, video, documents, and combinations of them. It is initialized from Gemma 4 (decoder reused as a bidirectional embedder) and trained for ~2.9T tokens through a staged curriculum, then checkpoint-merged and quantization-aware-trained to int4.

The paper's claim is a new Pareto frontier across embedding benchmarks in all modalities — reportedly beating specialist models and generalist models 2–12× larger. The architecture and deployment knobs matter more than any single eval win because they define the serving shape of a production embedding stack.

---

## Architecture at a glance

```
Backbone (271M, including embedder)
  24 transformer layers
  d_model = 512
  bidirectional attention (converted from Gemma 4's causal)
  mean pool → linear projection → L2 norm → 768-dim embedding
  per-layer embeddings: projection-only (no lookup table)   ← simplification vs Gemma

Vision encoder (167M, trainable)
  Vision transformer, patch size 16
  soft-token dim 768
  budget: 70 → 1120 tokens per image, default 280
         140 per video frame, up to 32 frames per video

Audio encoder (305M)
  2 downsampling conv layers + 12 Conformer layers (USM-based)
  40 ms chunks → 25 soft tokens/second
  soft-token dim 1536

Config footprints at serving time
  text-only:       271M
  vision-language: 439M
  audio-language:  577M
```

Max training sequence length is 8192 tokens. Matryoshka truncation targets at 768 (full), 512, 256, 128.

---

## Training recipe

Full details in [multimodal-embedding-curriculum](../multimodal/multimodal-embedding-curriculum.md); summary here.

### Stage 1 — Initialization

Backbone taken from a pretrained Gemma 4 decoder:
- Attention switched causal → bidirectional.
- Per-layer embedding **lookup table removed**; only the per-layer projections remain. This is the "simplified per-layer embeddings" of the title claim.
- Mean pool, linear projection to 768, and L2 norm added as the embedding head.

Vision encoder initialized from a smaller Gemma 4 encoder. Audio encoder initialized from USM and **frozen** throughout.

### Stage 2 — Text pre-finetuning (~300B tokens)

Unsupervised query-target pairs from QA, sentence similarity, code retrieval, web search, multilingual. **Loss is mostly distillation** (teacher embeddings from Gemini Embedding 2), with a small contrastive term. Teaches the bidirectionally-adapted backbone to be an embedder.

### Stage 3 — Multimodal pre-finetuning (~2T tokens)

Images, audio, video, and documents across unimodal, cross-modal, and mixed-modal tasks, mostly with hard negatives. **Distillation-only** loss. Twice as many steps as Stage 2. This is where the shared cross-modal space forms.

### Stage 4 — Multimodal finetuning (~600B tokens)

Switch from distillation to **contrastive loss**, add **spread-out** (second-moment) regularizer and **Matryoshka Representation Learning (MRL)** across all truncation dims. Lower LR and batch size; higher-quality task-specific data (tool retrieval, audio captioning, medical imaging). Objectives include listwise Plackett–Luce extensions of the usual contrastive NCE, with in-batch, hard, and same-tower negatives.

### Stage 5 — Checkpoint merging

Unweighted average of the final finetuning checkpoint and an earlier checkpoint from the same trajectory. Related to model souping but trajectory-local.

### Stage 6 — Quantization-aware tail patch

Separate QAT stage after merging, **distilled from the frozen float checkpoint**. Backbone int4 (per-block or per-channel); vision encoder W8A8. Keeps most of the float quality — see numbers below.

---

## Deployment story

EmbeddingGemma 2 ships the model with four orthogonal knobs:

1. **Selective modality encoder loading.** If an application only needs text, load only the 271M backbone. Adding vision costs +167M at load time, not inference time.
2. **Adjustable vision token budgets.** 70–1120 tokens per image (default 280); 140 per video frame. Quality vs. latency on the vision side without retraining.
3. **Truncatable embedding dimensions.** 768 (full) → 512 → 256 → 128 via MRL, with sub-linear quality drop. Downstream vector-DB cost scales with dim.
4. **Int4 weights (via QAT).** Backbone quantized to int4 with the tail-patch QAT above. Vision encoder stays W8A8.

Combined, a text-only int4 deployment uses roughly 1/10 the memory of float vision+audio+text and loses little quality — see MTEB numbers below.

---

## Key results

Headline numbers (as reported):

| Benchmark | Metric | Score |
| --- | --- | --- |
| MTEB (Multilingual, v2) | Mean (Task) / Mean (Type) | 61.36 / 53.82 |
| MTEB (English, v2) | Mean (Task) / Mean (Type) | 68.46 / 63.86 |
| MTEB (Code) | Mean (Task) | 78.68 |
| MMEB-v2 (vision) | Mean (Task) | 59.01 (Image 57.28, Video 50.67, VisDoc 67.84) |
| MIEB (lite) | Mean (Task) / Mean (Type) | 60.17 / 65.67 |
| MAEB (audio) | Mean (Task) / Mean (Type) | 49.87 / 48.65 |
| MSEB native-audio Retrieval | MRR@10 overall | 69.54 |

**Int4 QAT loss on MTEB Multilingual Mean(Task):** 61.36 (bf16) → 60.96 (int4 per-block) → 60.84 (int4 per-channel). Essentially free.

### Ablations worth naming (MMEB-v2 overall)

- **Projection-only per-layer embeddings** 55.46 vs. **with lookup table** 55.07. The simplification doesn't hurt and marginally helps.
- **Finetuning stage** 55.46 vs. 52.28 with extra distillation — contrastive loss earns its keep.
- **Trainable vision encoder** 52.27 vs. **frozen** 48.18. Freezing audio works; freezing vision doesn't.
- **Vision budget at inference:** 56.15 at 280 tokens, 57.02 at 560. Diminishing returns above 280.

---

## What's actually novel

Grounded in the paper's own ablations, the specific contributions are:

- **Simplified per-layer embeddings.** The projection-only variant of Gemma's per-layer embedding scheme. Marginal quality win plus a parameter save.
- **Modality × objective progression.** Three distinct training stages (text distill → multimodal distill → multimodal contrastive) with a specific switch of loss function at the right time, documented via ablation. See [multimodal-embedding-curriculum](../multimodal/multimodal-embedding-curriculum.md).
- **Teacher-distillation covering hard negatives.** The distillation loss uses teacher *truncated* embeddings; hard negatives are included inside the distillation target rather than as a separate contrastive term during pre-finetuning.
- **Tail-patch QAT distilled from float.** Instead of QAT from scratch or straight PTQ, a short QAT patch after checkpoint merging. Nearly free int4.
- **Deployment knobs engineered in.** Selective encoder loading, token budgets, truncatable dims, and int4 weights are first-class rather than retrofits.

### What is *not* novel to this paper, despite its appearance in the stack

- **Matryoshka Representation Learning** — predates EG2 (Kusupati et al. 2022).
- **Model souping / checkpoint averaging** — predates EG2.
- **Contrastive NCE with hard negatives** — standard embedding-model recipe.
- **USM audio encoder, Gemma backbone** — Google's prior work.

---

## Why it matters

Embedding models are the retrieval backbone of RAG, agent memory, and search. Shipping an open, unified, multi-modal, deployment-flexible embedder with a clear training recipe moves the production baseline. The architecture and the curriculum are the readable, replicable parts — the deployment knobs are what makes it a real product rather than a benchmark artifact.

---

## Sources

- Paper: *EmbeddingGemma 2: Unified and Efficient Multimodal Representations* — Schechter Vera, Dua, Andonov, Smoot, et al. (Google DeepMind), 2026 — [arXiv:2610.37364](https://arxiv.org/abs/2610.37364)
- Related: *Matryoshka Representation Learning* — Kusupati et al., 2022 — the truncatable-dim foundation.
- Related: Earlier `EmbeddingGemma` (v1) release notes — Google DeepMind.
