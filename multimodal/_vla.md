# Vision-Language-Action Models (VLA)

*Taxonomy — a family of models that extend vision-language backbones to also produce embodied actions and, in the latest generation, future state predictions.*

**TL;DR:** VLAs are vision-language models with an action head — they consume images (or video) and language and produce continuous or tokenized robot actions. The family started with RT-2 / RT-X (imitation-learning-style, mostly closed models) and has moved through OpenVLA and π0 to a 2026 generation (Qwen-VLA, PhysBrain 1.5) that treats **language, action, and dense visual targets as three tokenized streams** trained jointly with next-token prediction. Modern VLAs pretrain heavily on **human interaction video** to sidestep the robot-data bottleneck.

**Related taxonomies:** n/a
**Depth files covered here:** (no depth files yet — this taxonomy is a stub anchor)

---

## The problem

Robot learning historically ran on domain-specific stacks: a perception module, a planner, a controller, each with its own model and its own data. This does not compose well to open-world tasks, does not benefit from web-scale pretraining, and does not transfer across robot embodiments.

VLAs bet that the same recipe that made vision-language models general — one big transformer trained on tokenized multimodal data with next-token prediction — works for embodied action, if action itself is tokenized and if pretraining data can be scaled from real-robot demonstrations to human video.

## The shared pattern

Every VLA in the modern family shares:

1. **A vision-language backbone** — a pretrained VLM (Qwen-VL, PaLI, LLaVA-style, etc.).
2. **Tokenized actions** — discrete tokens covering end-effector poses / joint states / velocities, or a continuous head that is trained jointly with the LM loss.
3. **Multimodal targets beyond language** — recent entries add *dense visual predictions* (future frames, depth, robot mask) as additional token streams.
4. **Human video pretraining** — task-centered episodes of human interaction, with recovered motion, are the largest source of embodied supervision; robot trajectories and simulation cover the fine-tuning stage.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| RT-2 / RT-X (no depth file yet) | Cast robot actions as text tokens over an existing VLM | Closed; limited embodiments | Historical foundation |
| OpenVLA (no depth file yet) | Open weights, action-tokenized VLM fine-tune | Small compared to frontier VLMs | Reproducible open-source baseline |
| π0 (no depth file yet) | Flow-matching action head grafted onto a VLM | Bespoke action head | Continuous-control regimes |
| Qwen-VLA (no depth file yet) | DiT-based action decoder on the Qwen VLM stack, embodiment-aware prompt conditioning | Bespoke decoder | Multi-embodiment manipulation + navigation from one model |
| PhysBrain 1.5 (no depth file yet) | Unifies language, action, and dense visual targets as three tokenized streams under one next-token loss; human-video pretraining | Very expensive pretraining | Best open-source score on the current embodied benchmark suites |

## How to choose

- **A minimum-friction open-source VLA baseline for research:** OpenVLA-family. Small, reproducible, well-documented.
- **Multi-embodiment manipulation and navigation from a single model:** Qwen-VLA. The embodiment-as-prompt trick is the practical win.
- **A frontier-competitive open VLA and a research target for the "action as tokens" recipe:** PhysBrain 1.5. Averages 72.5 across 28 embodied benchmarks, on par with proprietary GPT-6-Astra / Gemini 3.6 Flash on those benchmarks in the paper's report.

Stacking notes: an action head choice (tokens vs continuous vs diffusion vs flow-matching) is largely orthogonal to the pretraining data mix (human video, robot trajectories, sim). The 2026 direction is *toward tokens under one autoregressive loss* because it composes with next-token pretraining most cleanly.

## Adjacent but distinct

- **Foundation models for robot control that are not VLMs:** BEHAVIOR / RoboCat / Gato-style — related in spirit, different in that they do not necessarily start from a strong pretrained VLM.
- **Video world models:** [panoramic and streaming world models](README.md) predict future frames but do not necessarily produce actions. World models can be composed with VLAs (predict + act) but sit in a different family.
- **Pure VLMs:** vision-language without an action head. VLAs strictly extend them.

## Sources

- *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* — Google DeepMind — 2023.
- *OpenVLA* — Kim et al. — 2024 — open-weights VLA.
- *π0: A Vision-Language-Action Flow Model* — Physical Intelligence — 2024.
- *Qwen-VLA: Unifying Vision-Language-Action Modeling* — Qwen / Alibaba — 2026.
- *PhysBrain 1.5: From Vision-Language Models to Physical Foundation Models* — DeepCybo Team — 2026 — [arXiv:2609.14973](https://arxiv.org/abs/2609.14973).
