# Chain-of-Frame (CoF) Reasoning
*Depth — reasoning that unfolds through temporally connected generated video frames.*

**TL;DR:** A video-generation analogue of Chain-of-Thought. Instead of intermediate *tokens* carrying reasoning state, intermediate *frames* do — a video generator is trained to lay out its answer step-by-step across a temporal rollout, with explicit **visual reasoning tokens** and **textual reasoning tokens** injected to scaffold intermediate state. Proposed with OpenCoF-17K + Wan-CoF (ByteDance Seed / CUHK, 2026).

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [README.md](./README.md)

---

## What it is

CoF reframes video generation as *reasoning by imagined rollouts*: the generated frames themselves are the reasoning trace, not the final answer. To make this work with existing video-diffusion generators (Wan2.2-I2V-A14B here), the recipe adds two ingredients:

1. **A reasoning-video dataset** — OpenCoF-17K, spanning 11 task families (spatial reasoning, temporal reasoning, physics prediction, etc.), where the ground-truth video *is* a stepwise reasoning trace.
2. **Reasoning tokens** — special tokens injected alongside standard frame tokens, split into two kinds:
   - **Visual reasoning tokens** carry low-level visual cues (positions, geometry, motion trajectories).
   - **Textual reasoning tokens** carry high-level semantic priors (what the question is, what the target concept is).

The generator learns to allocate frames to reasoning steps and use the two token streams to organize spatial and temporal reasoning respectively.

## How it works

Training pipeline:

1. Curate OpenCoF-17K — task-family-labeled reasoning videos with explicit stepwise structure.
2. Fine-tune a base video-diffusion generator (Wan2.2-I2V-A14B) on OpenCoF-17K with the reasoning tokens as additional conditioning inputs.
3. At inference, the model receives the task prompt + optional first frame and rolls out the reasoning as a video.

Attention analysis in the paper reveals:

- **Visual reasoning tokens** dominate at **early denoising steps** and **shallow layers** — they anchor the geometric/spatial scaffolding of the rollout.
- **Textual reasoning tokens** dominate at **later denoising steps** and **deeper layers** — they enforce semantic consistency across the trajectory.

The two token streams thus have complementary roles across model depth and denoising time.

## Why it matters

- **A new reasoning substrate.** Text CoT is one substrate; CoF adds video as a second. For tasks that are inherently temporal or spatial (physics prediction, trajectory forecasting, embodied reasoning), rolling out frames is more natural than emitting text tokens.
- **Video generators as world models.** CoF is a concrete step toward using video generators as *reasoning engines* for tasks a language-only model cannot handle well.
- **Dataset + open model + ablation-heavy analysis.** OpenCoF-17K + Wan-CoF is exactly the package a subfield needs to bootstrap around.

## Gotchas & tricks

- **CoF-suitable tasks are a subset.** Not every reasoning task benefits — tasks with clean symbolic structure often do better in text.
- **Reasoning tokens are load-bearing.** Ablating either the visual or textual reasoning tokens drops performance; the two streams are not redundant.
- **Depth/timestep specialization.** Because visual and textual tokens live at different depths and denoising steps, naïvely fusing them into one stream loses the specialization.
- **Long rollouts still degrade.** As with text CoT, longer CoF rollouts amplify accumulated generation errors; the field will need CoF-analogues of self-consistency and verifier-guided decoding.

## Sources

- Paper: *OpenCoF: Learning to Reason Through Video Generation* — ByteDance Seed, CUHK MMLab, CUHK IMIXR, 2026 — https://arxiv.org/abs/2607.08763
- Related: *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* — Wei et al., 2022 — the textual reasoning-trace ancestor.
- Related: *Wan2.2-I2V-A14B* — the video-diffusion base model fine-tuned.
