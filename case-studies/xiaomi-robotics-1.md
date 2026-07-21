# Case Study: Xiaomi-Robotics-1

*A foundational Vision-Language-Action (VLA) model that ports the LLM pretrain/post-train recipe to robotics: pretrain on 100K+ hours of real-world manipulation trajectories with auto-labeled scene-state-transition captions, then post-train to align those capabilities with specific robot embodiments and imperative human task prompts. Sets new SOTA on RoboCasa365 (57.6% vs. prior 46.6%) and RoboDojo (20.07 vs. prior 13.07), with clean scaling curves on both data and model size.*

**Related concepts:** [vla-pretraining](../multimodal/vla-pretraining.md) · [_vla](../multimodal/_vla.md) · [mid-training](../pre-training/mid-training.md) · [_data-curation](../data/_data-curation.md) · [deepseek-v3 case study](deepseek-v3.md)

---

## What this is

Xiaomi-Robotics-1, released 2026 by Xiaomi Robotics (`mi-robotics@xiaomi.com`). A vision-language-action foundation model whose central claim is that VLA training can be scaled the way LLMs are scaled: a large *pretraining* pass on cheap, plentiful trajectory data with a language-based conditioning signal (auto-labeled scene state transitions), then a *post-training* pass that aligns those capabilities with imperative task prompts and specific robot embodiments.

Two capabilities it aims for: (1) follow diverse language instructions to perform mobile manipulation in *unseen* environments out-of-the-box, and (2) fine-tune efficiently on novel dexterous tasks with minimal downstream data. The paper argues both scale together — a stronger pretraining checkpoint yields better out-of-the-box performance *and* better post-training data efficiency.

---

## Two-stage training recipe

### Stage 1 — pretraining

- **Data.** 100K+ hours of real-world manipulation trajectories, collected via **UMI (Universal Manipulation Interface)** devices across a massive scale of environments and tasks. UMI is a wearable hand-held gripper enabling scalable, embodiment-decoupled data collection (from Zhao et al., Stanford — an external primitive Xiaomi is using at scale).
- **Auto-labeling pipeline.** A scalable pipeline that annotates trajectory clips with natural-language captions describing *scene state transitions* — "the mug is now on the shelf," "the drawer is open." These captions are the pretraining conditioning signal, giving rich, precise, embodiment-agnostic supervision that decouples what the model learns (how the world changes) from how a specific robot achieves it. See [vla-pretraining](../multimodal/vla-pretraining.md).
- **Objective.** Predict actions conditioned on the current observation + state-transition caption. Broad, generalizable action-generation without commitment to any specific task prompt format.
- **Scaling.** The paper reports clean scaling curves on both pretraining data and model size — improvements consistent with (and shaped like) LLM pretraining scaling.

### Stage 2 — post-training

- **Goal.** Bridge the descriptive-state-transition head learned in pretraining with (a) imperative task prompts humans naturally use ("pick up the cup and put it in the sink") and (b) the concrete kinematics and controller conventions of specific robot embodiments.
- **Mechanism.** Aligns the pretrained model with imperative instructions and per-embodiment control. Analogous to LLM instruction tuning: same capabilities, retargeted to how humans issue tasks and how the robot expects control signals.
- **Scaling transfer.** A stronger pretraining checkpoint delivers better post-training results in real-robot evaluations in unseen environments — the improvements from pretraining scale don't get lost when re-aligning to imperative prompts.

---

## Key results

- **RoboCasa365** — new SOTA at **57.6%** success rate, up from the prior best of 46.6% (**+11 points**).
- **RoboDojo** — new SOTA at average **20.07**, up from 13.07 (**+7 points**, ~54% relative).
- **Data efficiency for downstream fine-tuning** — high; the pretraining foundation makes complex dexterous tasks trainable from little demonstration data.
- **Scaling behavior** — consistent across data scale × model size × downstream evaluation. The two-stage recipe transfers scale from pretraining to real-robot post-training performance.

Code and model checkpoints slated for release.

---

## Why this is a case study

Prior VLA work (RT-2, OpenVLA, Qwen-VLA) typically trained a single unified stage or used relatively small trajectory corpora. Xiaomi-Robotics-1's contribution is the **decomposition itself** — treating VLA training as *pretraining on cheap, plentiful, language-conditioned data* + *post-training on the small, expensive, embodiment-specific data*, mirroring what worked for LLMs.

The two-stage recipe is only viable because the auto-labeling pipeline exists: without an embodiment-agnostic language-conditioning signal on trajectories, there's no way to separate "what the world does" from "how this robot acts on it." The scene-state-transition captioner is the enabling primitive. See [vla-pretraining](../multimodal/vla-pretraining.md) for the mechanism.

## Related depth files

- [vla-pretraining](../multimodal/vla-pretraining.md) — the state-transition-labeled trajectory pretraining recipe.
- [_vla](../multimodal/_vla.md) — Vision-Language-Action families taxonomy (RT-2, OpenVLA, Qwen-VLA, Xiaomi-Robotics-1, ...) and where they diverge on action head, embodiment strategy, and data source.
- [mid-training](../pre-training/mid-training.md) — the state-transition stage functions as a mid-training bridge between raw sensory data and imperative-task supervision.

## Sources

- Paper: *Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories* — Xiaomi Robotics, 2026 — [arXiv:2607.15330](https://arxiv.org/abs/2607.15330).
- Related primitive: *Universal Manipulation Interface* (UMI) — Chi, Xu, Feng, Wu, Zhao, et al., Stanford — the wearable data-collection device Xiaomi scales up.
- Benchmarks: RoboCasa365 (365 diverse manipulation tasks) and RoboDojo.
