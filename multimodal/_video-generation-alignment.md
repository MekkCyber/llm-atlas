# Video Generation Alignment

*Taxonomy — post-training and alignment methods for video generation models.*

**TL;DR:** Pretrained video generators (text-to-video, image-to-video, world models) produce fluent-looking clips but routinely drift from user intent, violate temporal coherence, and ignore physical/safety constraints. Alignment methods adapt a pretrained generator so it reliably satisfies these requirements. Four families dominate — SFT, self-training / distillation, preference- & reward-based, and inference-time — organised in a 2026 survey around an **implicit vs explicit alignment** axis.

**Related taxonomies:** [../post-training/_post-training.md](../post-training/_post-training.md) · [../post-training/_rewards.md](../post-training/_rewards.md)
**Depth files covered here:** (none yet — this taxonomy anchors future depth files for the family)

---

## The problem

Alignment in video generation faces constraints that image / text alignment do not:

- **Error accumulates over time.** A per-frame mismatch snowballs into drift across tens of seconds.
- **Motion–appearance coupling.** Appearance edits often alter motion, and vice versa.
- **Multi-objective tradeoffs.** Quality, intent-faithfulness, temporal coherence, safety, controllability all push the generator differently.
- **Thin temporal supervision.** Labels for "did this clip remain coherent for 10 seconds" are expensive; proxies are noisy.

Post-training strategies adapt the pretrained generator without redoing the base pretraining run.

## The shared pattern

All alignment methods inject a signal that tells the generator *this* kind of output is preferred. The axis the survey proposes is *how the signal is enforced*:

- **Implicit alignment.** The signal is baked into training data or targets. The model is never told why; it just sees better exemplars. (SFT, self-training, distillation.)
- **Explicit alignment.** The signal is a scalar reward or constraint attached to generated samples, and the generator is nudged against it. (Preference-based, reward-based, inference-time guidance.)

The two can combine (SFT-then-RLHF-style stacks), but they're useful as separate axes.

## Variants

| Family | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Supervised fine-tuning (SFT) | Train on high-quality curated clips | No fine-grained control; needs quality clip corpus | Baseline polish for a well-defined style/domain |
| Self-training / distillation | Student imitates a stronger teacher's distribution, or own filtered samples | Compounds teacher failures; bland diversity | When you have a strong teacher or verifier |
| Preference-based (DPO-style, RLHF) | Train on pairwise human/judge preferences | Expensive label collection; preference reward hackable | Stylistic alignment, aesthetic control |
| Reward-based RL | Scalar reward (quality metric, learned RM) + policy gradient | Reward hacking; dense reward design is hard | Specific measurable targets (e.g. prompt faithfulness, action correctness) |
| Inference-time guidance | Modify the sampling procedure (CFG variants, test-time optimisation) | Per-sample cost; cannot fix pretraining gaps | Fast iteration without re-training |

## How to choose

The survey's practical guidance maps neatly onto this axis:

- Have curated exemplars but no reward model → SFT.
- Have a strong teacher (closed or open) → distillation; add SFT polish.
- Have a reliable scalar quality/faithfulness metric → reward-based RL; prefer forward-process RL variants with localised reward routing (e.g. attention-guided placement, 2026).
- Have human preference data → preference-based; DPO-style when possible to avoid rollouts.
- Need control at generation time only → inference-time guidance; least invasive but limited ceiling.

Modern production stacks compose multiple: SFT → preference-based → inference-time guidance for release-ready models.

## Adjacent but distinct

- **Image-generation alignment** shares the family map but has far more mature benchmarks and reward models — borrow techniques, re-benchmark on video.
- **World-model learning** overlaps with video generation but is evaluated differently (prediction accuracy on held-out frames) and tends to use SFT + self-training rather than preference-based signals.
- **Video *understanding*** (not generation) is a separate problem whose post-training uses standard VLM recipes.

## Sources

- Paper: *Video Generation Models: A Survey of Post-Training and Alignment* — Li, Gu, Kulkarni, Im, Honarmand, Wang, Song, Du, Jiang, Zheng, Li, Tao, Fazli; ASU / Twitch / Stanford / eBay / NewsBreak / Microsoft / Columbia / USC / CMU, 2026 — [arXiv:2610.00812](https://arxiv.org/abs/2610.00812). The survey this taxonomy follows.
- Related: *Adaptive Reward Routing: Dynamic Multi-Reward Optimization for Joint Audio-Video Diffusion via Forward-Process RL* — Yang et al., Tencent / HKUST, 2026 — [arXiv:2609.37200](https://arxiv.org/abs/2609.37200). A reward-based RL contribution covered by this axis.

---

## Conventions

- **Filename:** `_video-generation-alignment.md` (leading underscore distinguishes taxonomy from depth files).
- **Folder placement:** `multimodal/` — the home of VLM / video generation / audio-vision work.
- **Scope:** post-training and alignment of generators; architectural variants and pretraining choices live elsewhere.
