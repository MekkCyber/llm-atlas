# LongLive-Plug
*Depth — once-for-all distillation that trains reusable capabilities as LoRAs for video diffusion.*

**TL;DR:** Video diffusion models get specialized downstream (world modeling, robotics, editing, long-video, multimodal gen), and each specialization typically repeats a distillation stage (few-step sampling, CFG caching, long-context correction). **LongLive-Plug** (Yang et al., 2026) distills these capabilities **once** as LoRA adapters on a base model and plugs them, **training-free**, into any compatible downstream model — verified across **54 downstream models**, 3 backbone families, and 8 task categories.

**Prereqs:** [README.md](./README.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

Video diffusion labs ship a stream of specialists from the same base: world simulators, robot video predictors, video editors, long-video generators. Every specialist typically runs its own distillation pass to recover deployment-friendly inference properties (few-step sampling, single-pass CFG, long-context autoregressive correction). This re-work is pure overhead — the distilled capability is a property of the *base's* function space, not the specialist's.

LongLive-Plug separates **capability distillation** from **task specialization**. Train each capability as a LoRA on the base; let specialists ride on it unchanged.

## How it works

Three capability LoRAs on the base:

1. **Single-pass CFG LoRA.** Trained at a fixed guidance scale, but exposes inference-time guidance control via its merge weight. Classifier-free guidance normally needs two forward passes (conditional + unconditional) — a single-pass CFG LoRA produces the guided velocity in one pass.
2. **Few-step sampling LoRA.** Standard few-step distillation (progressive / consistency-style) packaged as a LoRA rather than a full-model finetune.
3. **Long-context error-correction LoRA.** For autoregressive long-video generation, this LoRA corrects drift accumulated over hundreds of frames.

Deployment is training-free: compose the LoRA stack with any downstream specialization LoRA. The paper verifies compatibility even when downstream models add **conditioning branches** or **expand output channels** — the base-level capabilities still ride.

Combining the few-step LoRA with the CFG LoRA simultaneously preserves few-step generation and CFG controllability on downstream tasks.

## Why it matters

- **54 downstream models × 3 backbone families × 8 task categories** confirm reuse. One distillation per backbone family covers an entire specialization catalog.
- **Economic shift for open-source video stacks.** Specialists get few-step inference and long-video correction for free; distillation is amortized across the family.
- **CFG-as-LoRA is a general trick.** The idea of putting guidance control in a LoRA with inference-time scale knobs can extend beyond video — image diffusion with multiple guidance schedules, audio diffusion, etc.

## Gotchas & tricks

- **Compatibility is "verified", not guaranteed.** The claim is empirical on 54 models. New backbone families need re-testing — architectural changes (e.g. a new attention variant) may silently break the capability LoRA.
- **LoRA rank matters.** Capability LoRAs trained at low rank may interfere with downstream LoRAs that use higher ranks on the same layers. Rank budgeting across the stack is a real tuning dim.
- **Merge order.** The paper's recipe merges capability LoRAs first, then downstream specialization. Reverse order can shift CFG behavior.
- **Specialist retraining can still beat.** If a downstream specialist redistills with full access to its own distribution, it may outperform plug-and-play on hardest-case metrics. LongLive-Plug trades last-mile quality for massive engineering savings.

## Sources

- Paper: *LongLive-Plug: Once-for-All Distillation for Video Generation* — Yang, Wang, Huang, Chen, Zhang, Fu, Ma, Lin, Mao, Chu, Han, Chen, 2026 — [arXiv:2609.38154](https://arxiv.org/abs/2609.38154).
