# Digital teleoperation (RynnWorld-Teleop)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Replace physical robots during VLA data collection with a **generative video world model**. An operator's hand-pose stream drives a robot-centric video diffusion transformer (DiT) that synthesizes high-fidelity egocentric video from a single reference image; the pose stream itself becomes an **embodiment-agnostic action label** transferable to any target robot via standard retargeting. Runs at 40+ FPS on a single H100, and policies trained purely on generated data achieve zero-shot Sim2Real on dexterous bimanual tasks.

**Prereqs:** *(none — sits atop video-diffusion basics)*
**Related:** [README.md](./README.md)

---

## What it is

Vision-Language-Action (VLA) foundation models are bottlenecked by demonstration data: physical teleoperation binds every trajectory to specific hardware and workspaces. Scaling data collection means scaling robot fleets, operator time, and workspace footprint.

Digital teleoperation decouples the demonstration from the physical rig. An operator wearing a hand-pose sensor "teleoperates" a **simulated robot in a generated video world**. Everything downstream — the video record, the state trajectory, the action labels — comes from the model, not from a real robot.

## How it works

Three main components:

**1. Reference-image-conditioned video DiT with depth-aware skeletal conditioning.** Given a single reference image of a robot and workspace, plus the operator's hand-pose stream, the video DiT synthesizes egocentric video frames of the robot executing the corresponding motion. Depth-aware skeletal conditioning grounds the hand poses in 3D so the generated motion is physically plausible.

**2. Progressive human-to-robot training.** The DiT is first trained on human egocentric video (abundant), then progressively adapted to robot embodiments. This lets it inherit natural motion priors from human data before specializing to specific hardware — a data-efficiency win.

**3. Streaming autoregressive distillation.** The generative process is distilled from an iterative diffusion sampling loop into a single-pass autoregressive inference — this is what enables the 40+ FPS real-time claim on a single H100.

**Action labels.** The operator's pose stream is the ground-truth action label. It's **embodiment-agnostic**: standard retargeting maps it to whatever target robot's joint space, so the same "demonstration" becomes training data for any hardware.

## Why it matters

- **Removes the hardware bottleneck** of VLA data collection. If a video model can generate demonstrations at 40 FPS, one operator produces demos at a rate previously impossible with any physical rig.
- **Zero-shot Sim2Real works.** Policies trained purely on generated data transfer to real dexterous bimanual tasks. This is the load-bearing empirical claim — if it didn't hold, the whole framework would be a video demo, not a data engine.
- **Cross-embodiment scaling for free.** Retargeting the pose stream to different robots gives multi-embodiment data with no new collection.
- **Augments real data reliably.** Mixing digitally-teleoperated data with real data consistently improves success rates over real-only — even when real data is available, generated data is a net positive.

## Gotchas & tricks

- **The reference image is a strong bottleneck.** The generated video inherits the reference frame's lighting, workspace layout, and object arrangement. Diversity in reference images matters as much as diversity in operator inputs.
- **Depth-aware conditioning depends on a working depth estimator.** In workspaces where the depth prior is bad (transparent objects, mirrored surfaces), the generated motion drifts.
- **Distillation to one-pass inference costs quality.** The distilled model is fast but not identical to the full iterative-diffusion baseline. For final data-quality curation, running the slow full version on a subset might be worth it.
- **Sim2Real gap remains for contact-rich tasks.** The paper focuses on dexterous bimanual manipulation; contact-force ambiguity in generated video means the physics-of-touch signal is soft. Real-data augmentation helps here.
- **Not a robotics simulator.** Digital teleoperation makes *video*, not a physics-consistent world — you can't do inverse dynamics or contact modeling from the outputs. It's a data engine, not a control-training environment.

## Sources

- Paper: *RynnWorld-Teleop: An Action-Conditioned World Model for Digital Teleoperation* — Zhao, Li, Gong, et al., DAMO Academy Alibaba / HK Embodied AI Lab / CUHK / Hupan Lab / Ant Group, 2026 — [arXiv:2607.06558](https://arxiv.org/abs/2607.06558).
