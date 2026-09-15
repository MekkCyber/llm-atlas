# Vision-Language-Action (VLA) models

*Taxonomy — foundation models that map (image / video, natural-language instruction, robot state) directly to actions or action trajectories.*

**TL;DR:** VLAs extend the VLM stack with an action head (or an action decoder) and are trained on robot-trajectory data — sometimes mixed with human egocentric video, simulation, or vision-language data. The design space splits along **how vision conditions the action** (direct fusion vs latent bottleneck), **what the action head is** (discrete action tokens, DiT decoder, chunked SE(3) poses), and **how the reasoning trace is used** (implicit vs a supervised CoT trunk driving action).

**Related taxonomies:** *(no sibling taxonomies in this folder yet — VLM stack coverage in [multimodal/README](README.md))*
**Depth files covered here:** [latent-interface-training](latent-interface-training.md)

---

## The problem

A vision-language-action model has to (a) perceive the scene, (b) ground the instruction, (c) reason about goal state, and (d) emit executable robot actions. When trained end-to-end on demonstration data, models tend to exploit **task-irrelevant visual cues** that correlate with actions inside the training distribution — the *vision-action shortcut*. Under distribution shift (unseen camera pose, lighting, distractors), the shortcut collapses. VLA design choices are largely about controlling *how* visual information reaches the action head so the shortcut doesn't dominate.

## The shared pattern

Every VLA has three components:

1. **Vision-language encoder** — usually a pretrained VLM (Qwen-VL, Molmo, InternVL, PaLI) that produces image + text embeddings.
2. **Reasoning / action trunk** — the block that fuses embeddings with robot state and decides what to do. Can be the same as the VLM (implicit reasoning) or a separate module (explicit CoT trunk).
3. **Action decoder** — emits actions. Can predict discrete action tokens, continuous chunks (SE(3) poses over $H$ steps), or diffusion / flow-matching over an action distribution.

Training data mixes robot trajectories with human-video, simulation, VLN, and vision-language pretraining data.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Direct end-to-end VLA (Pi0, Pi0.5, OpenVLA-family) | Fuse vision + language into the action decoder in one stack | Vulnerable to vision-action shortcuts | High-quality on-distribution demonstrations |
| DiT-decoder VLA (Qwen-VLA-style) | DiT on top of the VLM, embodiment-aware prompt conditioning | Heavier decoder | Multi-embodiment, action-and-trajectory prediction |
| World-action models (WAMs — FAST-WAM, ImageWAM) | Predict future *world state* alongside actions | More parameters to learn | Long-horizon planning; benefits from world-model regularization |
| Reasoning-conditioned VLA (MobileVLA-R1 2.0-style; no depth file yet) | Explicit CoT trunk feeds the action decoder; RL on the reasoning trace | Requires CoT supervision + RL loop | Long-horizon mobile manipulation & navigation |
| [latent-interface-training](latent-interface-training.md) | Vision only feeds action through a pose-supervised latent bottleneck | Two-stage recipe adds training cost | Distribution-shift generalization; plugs into any of the above backbones |

## How to choose

- **Building a VLA from scratch, single embodiment.** Direct end-to-end fusion is the simplest baseline.
- **Need multi-embodiment.** Embodiment-aware prompt conditioning (DiT-decoder VLA) avoids per-robot fine-tuning.
- **Long-horizon mobile / navigation tasks.** Explicit reasoning-conditioned trunks + RL on reasoning-to-action consistency are winning as of 2026.
- **Concerned about distribution shift.** Stack [LIT](latent-interface-training.md) on top of whichever backbone you use — it's framework-agnostic and empirically transfers across four recent VLA/WAM stacks.

## Adjacent but distinct

- Pure VLMs (Qwen-VL, InternVL, PaliGemma) — vision-language without an action head. VLAs *use* a VLM as the perception trunk.
- Diffusion policies — action decoders using diffusion / flow-matching, orthogonal to the VLM choice. Common inside VLA action heads.
- World models trained purely for video prediction — not actionable unless coupled with an action head.

## Sources

- Paper: *Breaking the Vision-Action Shortcut: Latent Interface Training for Generalizable Robotics Foundation Models* — Lin, Shailesh, Luo, Duan, 2026 — [arXiv:2609.12641](https://arxiv.org/abs/2609.12641).
- Paper: *MobileVLA-R1 2.0: RL-Enhanced Reasoning for Mobile Robot Control* — Huang, Huang, Zhang, Yan, Tang, 2026 — [arXiv:2609.06251](https://arxiv.org/abs/2609.06251).
- Background: *OpenVLA* and *Pi0* — canonical direct end-to-end VLA baselines.
