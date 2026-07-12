# Chain-of-Frame (CoF)
*Depth — reasoning that unfolds through temporally connected video frames.*

**TL;DR:** Chain-of-Frame reframes reasoning so its intermediate steps are *depicted*, not narrated. Instead of a text CoT, the model generates a temporally coherent sequence of frames where each frame is the visual analogue of one reasoning step. Introduced in *OpenCoF* with a supervised dataset (**OpenCoF-17K**, 11 task families) and a video model (**Wan-CoF**) fine-tuned on it, plus an optional inference augmentation using explicit **visual** and **textual reasoning tokens** injected during denoising.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [_rl.md](../_rl.md)
**Related:** [prm.md](prm.md) · [orm.md](orm.md) · [../rlvr.md](../rlvr.md)

---

## What it is

A supervised-and-then-representational recipe for teaching video generators to *reason* rather than merely animate. Two levers:

1. **Data lever — OpenCoF-17K.** A reasoning-video dataset spanning 11 task families (physics prediction, causal inference, spatial planning, counterfactual completion, and so on). Each sample is a short video whose frames constitute a legible reasoning trajectory.
2. **Representation lever — reasoning tokens.** During denoising, explicit visual tokens (low-level spatial cues) and textual tokens (high-level semantics) are inserted alongside the noisy latents. Attention analysis shows the two token types contribute at *different* depths and denoising steps: visual tokens dominate early-step spatial reasoning, textual tokens dominate late-step semantic decisions.

## How it works

Fine-tune a video diffusion backbone (Wan2.2-I2V-A14B in the paper) on OpenCoF-17K under an image-to-video objective, teaching it to unroll a reasoning-shaped trajectory conditioned on an initial frame + task description. At inference, the model can optionally accept a pair of extra token streams: a visual reasoning token per frame (spatial anchor) and a textual reasoning token (semantic anchor). The extra tokens are integrated via cross-attention inside the DiT blocks; their contribution is tracked with an attention heat-map that reveals the depth-vs-timestep-vs-modality decomposition.

## Why it matters

- **Reasoning where the state is visual.** Long-horizon spatial, physical, and counterfactual reasoning has an awkward fit with text CoT — CoF puts the reasoning substrate in the modality that natively encodes those variables.
- **Opens a new axis of scaling.** The reasoning-video dataset is the missing prerequisite for future reasoning-video RL (rewards on frame-level correctness, not text answer strings).
- **Attention analysis is transferable.** The visual-vs-textual depth/timestep decomposition is a diagnostic other multimodal reasoning stacks can borrow.

## Gotchas & tricks

- Frame count trades reasoning depth against denoising cost — CoF benefits saturate around the paper's dataset frame-count budget.
- Visual reasoning tokens help early, textual reasoning tokens help late — mixing them uniformly across depth is worse than the depth-conditioned injection.
- CoF-style supervision does not remove the need for text CoT on tasks whose state is purely symbolic.
- Wan-CoF gains hold across four video-reasoning benchmarks; single-benchmark gains can mislead — always evaluate on the full suite.

## Sources

- Paper: *OpenCoF: Learning to Reason Through Video Generation* — ByteDance Seed / CUHK MMLab / CUHK IMIXR, 2026 — [arXiv:2607.08763](https://arxiv.org/abs/2607.08763).
- Dataset / model: OpenCoF-17K and Wan-CoF (open-sourced with the paper).
- Background: text CoT origin — Wei et al., 2022 (Chain-of-Thought Prompting).
