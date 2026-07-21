# Vision-Language-Action Models (VLA)

*Taxonomy — foundation models that take vision + language and emit low-level robot actions.*

**TL;DR:** VLA models generalize the Vision-Language Model (VLM) recipe by adding an *action* modality — the model outputs low-level robot control signals (joint torques, end-effector poses, or discretized action tokens) conditioned on visual observations plus language instructions. The design space splits along three axes: **action head** (discretized tokens vs. continuous regression vs. diffusion decoder), **embodiment strategy** (single embodiment vs. embodiment-conditioned generalist), and **training data source** (human video vs. teleoperated demonstrations vs. embodiment-decoupled devices like UMI).

**Related taxonomies:** none yet in the graph.
**Depth files covered here:** [vla-pretraining](vla-pretraining.md)

---

## The problem

Robots need policies that translate high-level goals ("clean the table") into low-level actions (joint trajectories, gripper commands). Classical robotics builds these policies per-task, per-embodiment; scaling to open-world manipulation requires *foundation* policies that transfer across tasks and embodiments. VLMs give a great starting point — they already understand images and language — but they lack an action modality. VLA is the design space of "how do we bolt an action modality onto a VLM," and the choices matter.

## The shared pattern

All VLA models compose:

1. **A vision encoder** (SigLIP / CLIP / DINOv2-style) — turns observations into visual tokens.
2. **A language backbone** (often an LLM or VLM) — reasons over visual + language tokens.
3. **An action head** — converts the backbone's latent output into robot actions.

They differ on the last piece — the action head is where the design choices live — and on **what data** they were pretrained on and **how many embodiments** they target.

## Variants

| Model | Action head | Embodiment strategy | Data source | When it wins |
| --- | --- | --- | --- | --- |
| RT-2 (2023) | Discretized action tokens (autoregressive) | Single embodiment (Google robots) | Web VQA + robot demos co-training | Simple; benefits from web knowledge transfer |
| OpenVLA (2024) | Discretized action tokens | Multi-embodiment (Open X-Embodiment) | Open X-Embodiment robot demos | Open-weights baseline; broad benchmark coverage |
| Qwen-VLA (2026, [daily-papers/2026-05-29.md](../daily-papers/2026-05-29.md)) | DiT-based action decoder + embodiment-aware prompt | Multi-embodiment (prompt-conditioned) | Robot traj + human egocentric video + sim + VLN | Multi-task unification (manipulation + navigation + trajectory prediction) |
| [Xiaomi-Robotics-1](../case-studies/xiaomi-robotics-1.md) (2026) | Not fully disclosed; two-stage recipe | Embodiment-decoupled pretrain + embodiment-specific post-train | 100K+ hours UMI trajectories + auto-labeled state-transitions | Scales the pretrain/post-train split; strong out-of-the-box + data-efficient fine-tune |

## How to choose

The modern default for a **research-scale** VLA baseline is OpenVLA — open weights, discretized action tokens, tractable to fine-tune. For a **generalist multi-embodiment** system, follow the Qwen-VLA / Xiaomi-Robotics-1 pattern: prompt-conditioned embodiment plus a two-stage or heterogeneous pretraining recipe.

- **Discretized action tokens** — simplest, matches the LLM decoder, but coarse control. Fine for high-level pick-and-place, painful for dexterous manipulation.
- **DiT / continuous action decoder** — smoother control, better for dexterous / high-frequency tasks, extra inference cost.
- **Embodiment-conditioned prompt** — beats separate-head-per-embodiment for generalists, but only works when the corpus covers the target embodiment.
- **UMI-style embodiment-decoupled pretraining** (Xiaomi-Robotics-1) — currently the most scalable data source, but requires a post-training stage to bind the model to a specific robot.

## Adjacent but distinct

- **VLMs (vision-language models)** — same first two stages, no action head. VLA extends this by adding actions.
- **World models** — predict future states from actions, don't emit actions directly. Complement VLA (a VLA can plan against a world model).
- **Behavior cloning / imitation learning** — classical baseline. Modern VLA is BC scaled up with a VLM backbone and richer data.

## Sources

- Paper: *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* — Brohan et al., Google DeepMind, 2023.
- Paper: *OpenVLA: An Open-Source Vision-Language-Action Model* — Kim et al., 2024.
- Paper: *Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories* — Xiaomi Robotics, 2026 — [arXiv:2607.15330](https://arxiv.org/abs/2607.15330).
- Paper: *Qwen-VLA: Unifying Vision-Language-Action Modeling* — Qwen, 2026 (see [daily-papers/2026-05-29.md](../daily-papers/2026-05-29.md)).

---

## Conventions

- **Filename:** `_vla.md` (leading underscore for taxonomy in `multimodal/`).
- **Depth files reciprocate:** [vla-pretraining](vla-pretraining.md) links back via its `Related:` line.
- **Scope:** VLA foundation models. Task-specific single-embodiment robotic policies are out of scope — those belong in application literature, not this taxonomy.
