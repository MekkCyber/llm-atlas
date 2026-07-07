# Vision-Language-Action (VLA) Models
*Taxonomy — foundation-model-style models that emit low-level actions for embodied control.*

**TL;DR:** A Vision-Language-Action (VLA) model treats robot control as a token-emission problem: a VLM backbone consumes visual observations and a language instruction, and an action head emits low-level actions (joint torques, gripper commands, base velocities). The design axes are the *backbone* (VLM vs. dedicated encoder), the *action head* (regression, diffusion, autoregressive tokens, DiT), the *action horizon* (single step vs. chunked), and the *runtime* (Python research stack vs. optimized inference). Modern VLAs (RT-2, OpenVLA, π0, Qwen-VLA) mostly agree on VLM backbone + chunked action head; disagreement is in how to keep the chunk closed-loop, how to compress inference cost, and how to portably deploy across heterogeneous robots.

**Related taxonomies:** [../multimodal/README.md](../multimodal/README.md), [README](README.md)
**Depth files covered here:** [action-chunk-correction](action-chunk-correction.md) · [../inference/embodied-runtime.md](../inference/embodied-runtime.md)

---

## The problem

Robots need policies that generalize across scenes, embodiments, and tasks. Classical control policies do neither. The intuition behind VLA: reuse the same VLM stack that already generalizes over image + text tasks, and re-target the output head from language tokens to action tokens (or action-shaped continuous outputs). Same recipe, new modality on the output side.

The constraint that VLA-specific tricks are fighting: robots run *closed-loop*. Actions are executed in the physical world; the next observation depends on the action. An open-loop chain of 16 planned actions is fast but wrong the moment the world diverges from the model's prediction. A per-step re-plan is closed-loop but too slow for reactive control (100 Hz+ on manipulation, hundreds of Hz on locomotion).

Add: embodiments differ (arm morphology, gripper type, wheel base), edge hardware is heterogeneous (Jetson, custom ASICs, laptops), and the research-Python stack does not run on any of them well.

## The shared pattern

Every VLA has four blocks:

1. **Vision encoder** (SigLIP, DINO, VLM-internal ViT).
2. **Language conditioning** (instruction tokens, embodiment prompt).
3. **Backbone** (transformer, usually a VLM whose weights are inherited from a general-purpose pretrain).
4. **Action head** (regression, autoregressive action tokens, DiT-based action decoder, flow-matching action decoder).

VLAs differ in how each block is implemented and in what runtime executes them at deployment.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| RT-2 (Google, 2023) | Discretize actions into VLM vocab tokens; predict autoregressively | Requires action tokenization; limits precision | Simple to slot into any VLM decoder |
| OpenVLA (2024) | Open reproduction of RT-2 pattern with Llama-3 backbone | Open-source ceiling on capability | Community baseline |
| π0 / π0.5 (Physical Intelligence, 2024–25) | Flow-matching action head; continuous actions from language-conditioned VLM | New head, but continuous actions | Fine manipulation |
| Qwen-VLA (Alibaba, 2026) | Unified manipulation + navigation with DiT action decoder | One model across embodiments | Multi-embodiment production |
| [action-chunk-correction](action-chunk-correction.md) (VLA-Corrector, 2026) | Monitor head + online gradient guidance to fix mid-chunk drift | Test-time compute overhead | Contact-rich manipulation with chunked planning |
| [../inference/embodied-runtime.md](../inference/embodied-runtime.md) (Embodied.cpp, 2026) | C++ multi-rate runtime abstracting VLA/WAM behind five plugin layers | Systems complexity | Edge deployment on heterogeneous robots |

## How to choose

- **Precision matters (fine manipulation):** flow-matching or diffusion action heads (π0-style) beat discretized-token heads.
- **Multi-embodiment coverage:** DiT-style action decoders with embodiment-prompt conditioning (Qwen-VLA) trade some per-robot precision for one model across robots.
- **Closed-loop reactivity:** short chunks + [action-chunk-correction](action-chunk-correction.md) recover reactivity without discarding the throughput benefit of chunked decoding.
- **Deployment on edge:** [../inference/embodied-runtime.md](../inference/embodied-runtime.md) (or an equivalent C++/CUDA runtime) — Python-based research stacks fail at edge.
- **Research prototyping:** OpenVLA remains the cheapest starting point.

## Adjacent but distinct

- **World-action models (WAMs)** — instead of policy(obs) → action, learn world_model(obs, action) → next_obs. Closely related; share the same runtime story. Embodied.cpp targets both.
- **Vision-language navigation (VLN)** — a subset (navigation-only) that predates unified VLAs. Qwen-VLA treats VLN as one of its unified tasks.
- **Behavior cloning / imitation** — the training signal for many VLAs, but not a modeling axis on its own.
- **Classical MPC** — non-learned, model-based; complementary at the reactive control layer.

## Sources

- Survey ground: RT-2 (Brohan et al., 2023), OpenVLA (Kim et al., 2024).
- π0 / π0.5 — Physical Intelligence, 2024–2025.
- Qwen-VLA — Alibaba Qwen, 2026 — see the 2026-05-29 daily-papers digest.
- Embodied.cpp — Xu et al., 2026 — [arXiv:2607.02501](https://arxiv.org/abs/2607.02501).
- VLA-Corrector — Pan et al., 2026 — [arXiv:2607.01804](https://arxiv.org/abs/2607.01804).
