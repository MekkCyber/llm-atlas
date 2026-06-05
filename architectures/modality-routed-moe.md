# Modality-routed MoE
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Modality-routed mixture-of-transformers partitions experts by *modality* rather than by content. Each token's modality (text, image, video, audio, action) deterministically picks its expert pool; routing is therefore zero-compute and zero-router-loss. Shared global attention and cross-modal layers handle interaction. Used as the backbone for omnimodal world models like Cosmos 3 (NVIDIA, 2026).

**Prereqs:** [_moe.md](_moe.md), [transformer-block.md](transformer-block.md)
**Related:** [deepseek-moe.md](deepseek-moe.md), [load-balancing-loss.md](load-balancing-loss.md), [aux-loss-free-balancing.md](aux-loss-free-balancing.md)

---

## What it is

Standard MoE routes each token to top-$k$ experts via a learned router, with a balancing loss to prevent expert collapse. This is *content-routed* — the router learns to send "math" tokens to one expert, "code" to another, etc.

Modality-routed MoE replaces the learned router with a deterministic mapping: each token carries a modality label (assigned at the embedding layer), and that label selects which expert pool it visits. There is no top-$k$ scoring, no router gradient, no balancing loss; the routing is by construction.

This is the natural design when:

- the modality set is **fixed and known** at training time (text, image, video, audio, action — Cosmos 3's set);
- each modality has enough data to merit its own expert capacity;
- you want all-to-all attention between modalities but per-modality MLP capacity.

## How it works

A typical block:

1. **Attention layer** (shared across modalities). Tokens of all modalities attend to each other in one global self-attention; this is how cross-modal interaction happens.
2. **Modality-routed MLP.** Each token is dispatched to its modality's MLP expert (no router score, no scoring overhead). Different modalities can have different MLP widths if needed.
3. **Residual + layer norm.** Standard.

Variations:

- *Strict modality routing* — one expert per modality, deterministic.
- *Hybrid* — combine modality-routed and content-routed experts in the same block (content-routed shared experts plus modality-specific).
- *Mixed at certain layers* — early layers content-routed, deeper layers modality-routed (or vice versa).

Cosmos 3 (2026) uses a Mixture-of-Transformers backbone with this pattern, supporting both understanding (text/image/video/audio/action in) and generation (any modality out).

## Why it matters

- **Predictable routing.** No router collapse, no auxiliary balancing loss, no router-warmup phase. Modality labels are known at every step.
- **Per-modality capacity.** You can scale audio expert width independently of text expert width based on data and task budget.
- **Natural for omnimodal generative models.** When the input/output modality set is fixed (text↔image↔video↔audio↔action), content-routed MoE wastes effort learning what amounts to a modality classifier; modality routing skips the problem.
- **Composes with shared attention for cross-modal interaction.** Global self-attention still mixes modalities; modality routing only specializes the MLP.

## Gotchas & tricks

- **Modality imbalance in data.** If one modality dominates the training corpus, its expert sees most of the compute. Up-sampling or per-modality learning-rate tuning may be needed.
- **Adding a modality is expensive.** Unlike content-routed MoE (where adding a new domain is just more data), adding a new modality requires a new expert and re-balanced training mix.
- **Cross-modal alignment relies on attention layers.** The shared-attention design must be deep enough to learn cross-modal correspondences; otherwise modality experts become silos.
- **Doesn't replace content routing for within-modality specialization.** Text alone may still benefit from content-routed experts on top of modality routing.

## Sources

- Paper: *Cosmos 3: Omnimodal World Models for Physical AI* — NVIDIA, 2026 — [arXiv:2606.02800](https://arxiv.org/abs/2606.02800).
- Related: Mixture-of-Transformers literature; content-routed MoE for contrast.
