# Omni-Modal Embedding
*Depth — turning a pretrained omni-modal LLM into a universal (text/image/video/audio) embedding model.*

**TL;DR:** Rather than assembling modality-specific towers, take a pretrained omni-modal LLM (Qwen-omni) as the shared backbone and contrastively adapt it into a universal embedder. Three ingredients: native omni initialization via low-rank contrastive tuning, homogeneous-source batch sampling (task-consistent batches → informative in-batch negatives), and focal-loss + similarity-based embedding distillation from expert teachers. Introduced by Ovis-Embedding (Alibaba, 2026).

**Prereqs:** [README.md](./README.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)

---

## What it is

A recipe for building a universal embedding model that natively handles text, image, video, and audio in a shared representation space, by contrastively fine-tuning a pretrained multimodal LM rather than fusing single-modality encoders. The whole approach hinges on the observation that a multimodal LM has already learned cross-modal alignment during pretraining — the embedding tuning stage just has to specialize it for retrieval.

Distinct from CLIP-style dual-tower setups: one shared backbone processes all modalities; there are no per-modality projection heads at the encoder level. Distinct from independent modality embedders + late fusion: modalities live in one space from the first layer.

## How it works

**1. Native omni-modal initialization.** Start from Qwen-omni pretrained weights. Add a low-rank adapter (LoRA-style) and train contrastively — the LoRA lets the model adapt to the embedding objective without wrecking the pretrained cross-modal knowledge in the base.

**2. Data-centric omni-modal training.** Assemble a broad corpus covering text, image, video, audio, and *interleaved* multimodal data (e.g. text+image, video+audio). Contrastive tuning needs in-batch negatives to work; the paper introduces **homogeneous-source sampling** — build batches so all items in a batch come from the same task/source, producing more informative near-miss negatives than random batching. Focal loss on top up-weights hard examples.

**3. Similarity-based embedding distillation.** Auxiliary expert embedders (specialized per-modality models) provide fine-grained similarity structure; distill their pairwise similarity matrices into the shared model. This transfers what modality specialists learn without adding inference-time modality towers.

**4. Inference-time low-rank feature decomposition.** Applies SVD-style decomposition to the embedding to give flexible dimensionality — shorter embeddings for fast retrieval, longer ones when accuracy matters — with minimal loss.

## Why it matters

- **Modality fragmentation tax.** Independent per-modality embedders + late fusion require alignment layers, per-pair training and separate deployment. A single-backbone omni model avoids all three.
- **State-of-the-art across benchmarks.** MMEB-v3/v2, MVEB, MAEB, RTEB — all reported as SOTA in the paper.
- **Reuses pretraining investment.** Contrastive tuning of an existing multimodal LM is cheap relative to training a multimodal backbone from scratch.
- **Any-to-any retrieval.** Because modalities share a space, retrieval across arbitrary modality pairs (audio→image, video→text) is free.

## Gotchas & tricks

- **Backbone choice matters more than the contrastive recipe.** Ovis-Embedding hinges on Qwen-omni being a strong multimodal LM to begin with. Applying the same recipe to a weaker omni base gives less.
- **Homogeneous-source sampling is the load-bearing data trick.** Random batching underperforms because most negatives are trivial. Grouping by source concentrates hard cases in each batch.
- **Focal loss + hard-negative sampling interact.** Both up-weight hard examples; too much of both destabilizes training. Tune together.
- **Low-rank inference decomposition is post-hoc.** Retrieval at reduced dimensionality is not free — some hard queries need the full-dimension embedding. Measure recall@k as a function of embedding dimension.

## Sources

- Paper: *Ovis-Embedding: Pushing the Frontiers of Universal Omni-Modal Embeddings* — Ovis-Embedding Team (Alibaba), 2026 — [arXiv:2609.25165](https://arxiv.org/abs/2609.25165).
