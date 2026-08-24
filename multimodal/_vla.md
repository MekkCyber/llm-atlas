# Vision-Language-Action Models

*Taxonomy — foundation models that take images + language in and emit robot actions out.*

**TL;DR:** VLA models extend the VLM stack with an action head, so a single network conditions on natural-language instructions and visual observations to produce motor commands. The design space splits along (1) action-head architecture (discrete tokens, continuous regressor, diffusion / flow decoder), (2) how much pretrained VLM is preserved, and (3) whether planning is single-shot or hierarchical (with a slow planner and a fast controller). The modern trend is *hierarchical* VLAs with generative action heads and, increasingly, test-time compute at the planner.

**Related taxonomies:** [../post-training/_rl.md](../post-training/_rl.md)
**Depth files covered here:** [vla-test-time-compute](vla-test-time-compute.md)

---

## The problem

Robots need policies that generalize across scenes, embodiments, and language goals. Bespoke per-task controllers do not; VLMs do, but do not emit actions. A VLA is the attempt to keep VLM generalization while adding a usable control interface, without either (a) collapsing all embodiments into one action space or (b) retraining the vision backbone from scratch per robot.

## The shared pattern

Every VLA has three parts:

1. **Vision-language backbone** — a pretrained VLM (typically frozen or lightly tuned). Provides scene understanding and language grounding.
2. **Action decoder** — a head that maps the VLM's hidden state to motor commands. Design choices: token-quantized actions, MLP regressor, diffusion/flow head.
3. **Data recipe** — heterogeneous robot data (multi-embodiment trajectories) plus grounding data (VLM training + egocentric video + simulation).

The shape of the whole system — how much control policy is inside the network vs outside, and how many hierarchical layers exist — is what distinguishes the variants.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| RT-2 (Google, 2023) | Discretize actions into VLM vocabulary tokens; VLM autoregresses actions | Discretization loses precision; fast inference | Simple manipulation, action grammar small |
| OpenVLA (2024) | Open version of RT-2 with a stronger vocab and multi-embodiment training | Same discretization limit | Community-scale VLA research |
| π₀ / π-flow (Physical Intelligence, 2024) | Continuous action head via flow matching on top of VLM features | Slower action generation; high fidelity | Dexterous, high-frequency control |
| RDT-1B (2024) | Diffusion transformer as the action head | Extra sampling steps at inference | Rich action distributions, multimodal actions |
| τ₀-VLA (2026) — see [vla-test-time-compute](vla-test-time-compute.md) | Hierarchical + world-model-guided TTC at the planner | Requires a good learned world model | Long-horizon manipulation with hard decisions |

## How to choose

- **Fast, simple manipulation** → discrete-token action head (RT-2 / OpenVLA lineage). Cheapest at inference.
- **Dexterous, continuous control** → generative action head (flow matching or diffusion). Pay in latency for smoother, more precise trajectories.
- **Long-horizon, multi-stage tasks** → hierarchical VLA with a planner and a controller. Consider adding test-time compute at the planner if a decent world model exists.
- **Zero-shot across novel embodiments** → embodiment-aware prompt conditioning + broad multi-embodiment pretraining beats per-robot fine-tuning.

Combining works: a hierarchical VLA can use a flow-matching action head at the low level and world-model TTC at the high level.

## Adjacent but distinct

- **VLMs** ([../multimodal/README.md](README.md)) — same backbone family but no action head. Everything below the action decoder is shared.
- **Model-based RL** — VLA + world model looks like MBRL, but VLA world models are typically pixel-space video models, not compact latent dynamics.
- **End-to-end driving stacks** — similar pattern (perception + action), but usually not language-conditioned and rarely built on general VLMs.

## Sources

- Paper: *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* — Brohan et al., 2023.
- Paper: *OpenVLA: An Open-Source Vision-Language-Action Model* — Kim et al., 2024.
- Paper: *π₀: A Vision-Language-Action Flow Model for General Robot Control* — Physical Intelligence, 2024.
- Paper: *τ₀-VLA: A Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation* — 40-author team, 2026 — [arXiv:2608.16885](https://arxiv.org/abs/2608.16885).
