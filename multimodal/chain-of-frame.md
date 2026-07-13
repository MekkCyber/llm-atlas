# Chain-of-Frame (CoF) Reasoning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The **video-generation analogue of Chain-of-Thought**: instead of tokens spelling out an intermediate reasoning trace, **temporally-connected frames** produced by a video generator serve as the reasoning steps. The paper introduces the CoF framing, an open dataset (**OpenCoF-17K**, 11 task families) that supplies the temporal supervision missing from general video corpora, and a fine-tuned model (**Wan-CoF** on top of Wan2.2-I2V-A14B). Also introduces explicit **visual and textual reasoning tokens** that capture low-level visual cues and high-level semantic priors as separate channels across denoising steps.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](./../post-training/reasoning/long-cot-rl.md)
**Related:** [../post-training/reasoning/prm.md](./../post-training/reasoning/prm.md) · [../multimodal/README.md](./README.md)

---

## What it is

CoF reframes video generation as reasoning. A video generator, when prompted with a physical-plausibility or logical-consequence task, produces frames that *are* the reasoning — the model must literally show what happens next, not describe it. This makes video generators a candidate reasoning substrate distinct from text CoT, with different failure modes and different training data needs.

Vanilla video generators are trained on general web-scale corpora that contain plenty of motion but very little *reasoning* supervision — cause-and-effect chains, physical consequences, task completions. OpenCoF is a targeted dataset (17K clips, 11 task families) that fills that gap.

## How it works

**OpenCoF-17K dataset.** 17K reasoning clips spanning 11 task families. Each clip carries dense frame-level supervision that captures a small piece of causal/physical reasoning. This is what makes CoF learnable rather than a framing exercise.

**Wan-CoF fine-tuning.** Start from Wan2.2-I2V-A14B (a strong image-to-video base). Fine-tune on OpenCoF-17K. Evaluate across four video reasoning benchmarks; report considerable gains over the base.

**Visual and textual reasoning tokens.** Beyond dataset fine-tuning, the paper adds explicit reasoning tokens to the model:

- **Visual reasoning tokens** capture low-level visual cues (edges, object positions, small dynamic details) across the frame sequence.
- **Textual reasoning tokens** capture high-level semantic priors (task descriptions, action goals) as a separate channel.

Attention analysis in the paper shows the two token types play distinct roles across model depth, denoising step, space, and time — visual tokens dominate spatial reasoning at fine denoising steps, textual tokens dominate high-level temporal planning at coarse steps.

## Why it matters

- **New reasoning substrate.** Text CoT has become the frontier's default reasoning trace. CoF opens a parallel: reasoning that lives natively in the *modality of the world* (frames), which for tasks like physical simulation, agent action planning, and world modeling may be closer to the right representation than words.
- **Open baselines.** Dataset + model + code are all released. That gives the community a shared floor for reasoning-oriented video generation, comparable to what OpenLongCoT did for text reasoning traces.
- **Temporal supervision as the missing ingredient.** The paper's message isn't "bigger video model" — it's "the specific temporal-reasoning signal is what generic corpora lack." Small, curated reasoning datasets over strong bases beat scaling up general video.

## Gotchas & tricks

- **CoF ≠ world model.** A CoF reasoner produces frames that reason; it isn't (yet) a full physics-consistent world model. Physical consistency across long horizons remains open.
- **Reasoning tokens interact with denoising schedule.** Because visual/textual tokens' roles shift across denoising steps, changing the sampler schedule after fine-tuning can degrade CoF quality. Refine schedule and tokens together.
- **17K is small for video.** Wan-CoF's gains are on top of a strong base; training a CoF model from scratch on 17K clips would not work. This is a fine-tuning recipe, not a from-scratch one.
- **Evaluation is nascent.** The four benchmarks used are recent and small; expect the field to build harder ones over the next year. Numbers should be read as "considerable gains vs. base" more than "SOTA that will hold."

## Sources

- Paper: *OpenCoF: Learning to Reason Through Video Generation* — ByteDance Seed / CUHK MMLab / CUHK IMIXR, 2026 — [arXiv 2607.08763](https://arxiv.org/abs/2607.08763).
- Base model: *Wan2.2-I2V-A14B* — public image-to-video base used for Wan-CoF.
- Related concept: *Chain-of-Thought for text reasoning* — Wei et al., 2022 — the reasoning-trace framing CoF generalizes to frames.
