# Knowledge distillation for LLMs
*Taxonomy — training a smaller student to imitate a larger teacher's outputs.*

**TL;DR:** Small language models are how most orgs ship LLMs, and distillation is how you recover a small model that behaves like a large one. The design space spans *what the student imitates* (final answers, top-K logits, full vocabulary logits, hidden states) × *when the teacher runs* (online, offline / cached) × *how the loss is computed* (KL, MSE, hard-target CE). The 2026 wave of efficiency work (offline top-K, fused chunked KL) turned distillation from an expensive art into a cheap high-throughput recipe.

**Related taxonomies:** [../_post-training.md](../_post-training.md), [../_rewards.md](../_rewards.md)
**Depth files covered here:** [offline-top-k-distillation](offline-top-k-distillation.md) · [chunked-kl-loss](chunked-kl-loss.md)

---

## The problem

Serving a frontier-scale model at production latency and cost is impossible for most deployment targets. Training a small model from scratch to match frontier quality is impossible for most training budgets. Distillation splits the difference: use the frontier model as a teacher to shape the small student, so the small student inherits capabilities without needing frontier-scale pretraining.

## The shared pattern

Every variant defines:
1. **A teacher signal:** what the teacher exposes for the student to imitate (final tokens, top-K logits, full logits, intermediate states, chain-of-thought).
2. **A loss:** how student output is measured against the teacher signal (KL, MSE, CE against hard targets).
3. **A training-time coupling:** whether the teacher runs alongside the student (online), was precomputed (offline / cached), or was run once in a data-generation step (rejection-sampled traces).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Online full-vocab KL | Teacher runs every step; KL over full vocab | Highest fidelity; teacher pinned in memory | Small models where teacher fits |
| [offline-top-k-distillation](offline-top-k-distillation.md) | Cache teacher top-K logits once; train student against cache | Cache size / IO cost; ~identical loss to online | Any teacher too large to run online; many ablations |
| [chunked-kl-loss](chunked-kl-loss.md) | Fused per-chunk KL kernel; memory linear in seqlen | Chunk-size tuning; kernel complexity | Long-context distillation on single GPU |
| Rejection-sampled trace distillation (no depth file yet) | Sample teacher, keep correct trajectories, SFT student on them | Loses teacher's probability shape; hard-target only | When teacher is closed / API-only |
| Hidden-state distillation (no depth file yet) | Match teacher's intermediate activations | Requires matched architectures; expensive backprop-through-teacher | Tight matched-arch student/teacher pairs |
| Speculative-decoding distillation (no depth file yet) | Distill for the specific role of "draft model" | Narrow objective; not general capability | Serving-side speculative decoding pipelines |

## How to choose

The **modern default** for capability distillation of a compressed model is offline top-K KL + a fused chunked KL kernel — you get near-identical loss to online, minus the teacher-in-memory tax, with 4× the context ceiling. Reach for rejection-sampled trace distillation when the teacher is API-only (no logit access) — see [../rejection-sampling.md](../rejection-sampling.md). Reach for hidden-state distillation only when you control both models' architectures and can afford the backprop-through-teacher cost.

For domain-specific distillation (reasoning, code), pair KD with a downstream RL phase — distillation gives you the base capability shape, RL sharpens the verifiable-reward axis.

## Adjacent but distinct

- [../rejection-sampling.md](../rejection-sampling.md) — teacher generates traces, student trains on kept ones. A distillation cousin without soft-target logits.
- [../cot-reward-model.md](../cot-reward-model.md) — reward-model shaping of student reasoning; complementary to KD, not a replacement.

## Sources

- Paper: *Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss* — Ryskulov et al., Multiverse Computing, 2026 — arXiv:2608.03796.
- Foundational: *Distilling the Knowledge in a Neural Network* — Hinton, Vinyals, Dean, 2015.
- Practice: MiniLLM (Gu et al., 2023), DistiLM (2024), and many others cover online-KD variants.
