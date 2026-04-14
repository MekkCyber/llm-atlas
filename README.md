# How Models Are Trained

A structured knowledge wiki to help you understand, with your favorite LLM, how modern language models actually work — from architectures and distributed training to RL post-training, reasoning, inference, safety, and interpretability.

---

## Why This Exists

Most publicly available material on AI training falls into one of two categories: paper summaries that strip away implementation details, or tutorial code that never scales past a single GPU. Neither explains **how production AI systems actually work**.

This repository is a personal zero-to-hero reference for LLMs, organized so that any single topic can be read in isolation or followed as a learning path. Concepts are extracted from papers, blogs, and courses as they're read, and deposited in the folder they belong to. Milestone papers and end-to-end systems land in `case-studies/`.

---

## How It's Organized

- **Topic folders** hold concept pages — one concept per file, deduped across papers. When a new paper touches an existing concept, the concept file gets updated (not duplicated).
- **`case-studies/`** holds milestone papers and end-to-end systems. They link to concept pages instead of re-explaining them.
- **Each folder's `README.md`** gives a suggested reading order, turning "zero to hero" into an explicit path per topic.
- **Cross-topic links** are handled via a `Related:` line at the top of each concept file. No duplicate files across folders.
- **Two kinds of concept files**, distinguished by a one-line italic label under the title:
  - **Depth files** — one specific technique, grounded in its source paper(s). Follow [`TEMPLATE-DEPTH.md`](TEMPLATE-DEPTH.md). Named after the technique (`rope.md`, `gqa.md`, `adamw.md`).
  - **Taxonomy files** — overviews of a class of techniques, linking out to depth files. Follow [`TEMPLATE-TAXONOMY.md`](TEMPLATE-TAXONOMY.md). Named after the category (`positional-encoding.md`, `optimizers.md`, `attention-variants.md`).

---

## Topics

### Fundamentals
Tokenization, embeddings, positional encoding, optimizers, losses, normalization, activations, initialization, prompting. The prerequisites everyone assumes. → [`fundamentals/`](fundamentals/)

### Architectures
Transformers, state-space models (Mamba, S4), mixture-of-experts, attention variants (MLA, GQA), positional encodings. → [`architectures/`](architectures/)

### Pre-Training
Data pipelines, parallelism (FSDP, TP, PP), mixed precision, scaling laws, training stability, the decisions that shape a base model. → [`pre-training/`](pre-training/)

### Post-Training
RLHF, PPO, GRPO, DPO, reward modeling, preference learning, rollout generation — plus fine-tuning and reasoning as subfolders. → [`post-training/`](post-training/)
  - Fine-tuning → [`post-training/fine-tuning/`](post-training/fine-tuning/)
  - Reasoning → [`post-training/reasoning/`](post-training/reasoning/)

### Systems (Training Infra)
Ray, distributed scheduling, fault tolerance, checkpointing, orchestration, rollout workers — the training-time backbone. → [`systems/`](systems/)

### Inference
KV cache, paged attention, continuous batching, speculative decoding, vLLM/SGLang, prefill/decode disaggregation — the serving-time stack. → [`inference/`](inference/)

### Quantization
FP8, INT4, MXFP4, GPTQ, AWQ, calibration, mixed-precision tradeoffs. → [`quantization/`](quantization/)

### Data
Curation, filtering, deduplication, tokenization strategy, mixtures, synthetic data. → [`data/`](data/)

### Multimodal
Vision encoders, projection layers, multimodal pretraining and post-training, video. → [`multimodal/`](multimodal/)

### Agents & Tool Use
Tool calling, function schemas, multi-turn execution, MCP, training with environments. → [`agents/`](agents/)

### Evaluation
Benchmarks, reward model evaluation, pairwise comparison, contamination. → [`evaluation/`](evaluation/)

### Safety & Alignment
RLHF for safety, constitutional AI, red-teaming, jailbreaks, refusal training, dangerous capability evals. → [`safety/`](safety/)

### Interpretability
Probing, mechanistic interpretability, circuits, SAEs, activation steering, logit lens. → [`interpretability/`](interpretability/)

### Case Studies
End-to-end breakdowns of real systems — Composer 2, DeepSeek R1, and more. → [`case-studies/`](case-studies/)

---

## Repository Structure

```
how-models-are-trained/
├── README.md
├── READING-LIST.md             ← 100-paper reading list
├── TEMPLATE-DEPTH.md           ← depth-file template (one technique)
├── TEMPLATE-TAXONOMY.md        ← taxonomy-file template (overview)
├── _assets/                    ← diagrams and figures
├── fundamentals/
├── architectures/
├── pre-training/
├── post-training/
│   ├── fine-tuning/
│   └── reasoning/
├── systems/
├── inference/
├── quantization/
├── data/
├── multimodal/
├── agents/
├── evaluation/
├── safety/
├── interpretability/
└── case-studies/
```

**Navigation:**
- Each topic folder's `README.md` has a suggested reading order — start there for a topic.
- Depth files follow [`TEMPLATE-DEPTH.md`](TEMPLATE-DEPTH.md) — TL;DR, mechanism, why it matters, gotchas, sources.
- Taxonomy files follow [`TEMPLATE-TAXONOMY.md`](TEMPLATE-TAXONOMY.md) — problem, variants-at-a-glance table, how to choose.
- Case studies link to concept pages rather than duplicating content.

---

## Workflow

1. Read a paper / blog / course.
2. Extract the important concepts.
3. Drop each concept into the folder it belongs to (create the file if new, update if it exists).
4. If the paper is a milestone or end-to-end system, add a case study that links to the relevant concept pages.
5. Use [`TEMPLATE-DEPTH.md`](TEMPLATE-DEPTH.md) for a specific technique, [`TEMPLATE-TAXONOMY.md`](TEMPLATE-TAXONOMY.md) for an overview of a class of techniques.

---

## Audience

Written primarily for me, but usable by:
- **ML engineers** building or debugging training pipelines
- **Infrastructure engineers** working on training or serving systems
- **Researchers** who want the full stack, not just the algorithm
- **Anyone prepping for AI research interviews** who wants the zero-to-hero map in one place

Assumes Python, some PyTorch, and a conceptual grasp of how neural networks are trained.

---

## License

MIT
