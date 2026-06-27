# In-Context World Modeling (ICWM)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A Vision-Language-Action model fails to generalize to new camera angles, robot morphologies, or calibrations because it conditions only on the current observation + instruction — implicitly assuming the training-time execution context. **ICWM** instead lets the policy run **a short sequence of self-generated, task-agnostic probes** at the start of an episode and reads those probes from its context window. The probes carry the system identification; the same frozen weights adapt to novel setups with no parameter updates.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [visual-tokenization](visual-tokenization.md), [../post-training/on-policy-distillation.md](../post-training/on-policy-distillation.md)

---

## What it is

A run-time adaptation technique for VLAs. Classical in-context learning (ICL) uses few-shot demos in the prompt to say *what task* to perform. ICWM uses the context window for a different purpose — to encode *how the system operates* — by ingesting a short prefix of the agent's own probing interactions before task execution.

Concretely, the policy is trained to:

1. **Emit task-agnostic probing actions** at the beginning of an episode (e.g., small calibration moves) without being told a task yet.
2. **Read its own probe observations + actions back in** as the context preamble for the rest of the episode.
3. **Implicitly infer system variables** — camera intrinsics/extrinsics, end-effector calibration, dynamics — from the preamble and condition all subsequent action predictions on it.

At deployment time, a novel setup (new camera viewpoint, swapped robot arm) is handled by running the same probing routine on the new hardware; no fine-tuning is needed.

## How it works

Three pieces of training:

1. **Probing-action curriculum.** The policy learns to emit short, informative probes that span the system's dynamics dimensions of variation. Generated on-policy, not scripted.
2. **Long-context conditioning.** During training the policy is conditioned on its own probe sequences from *different* simulator system configurations. This forces the weights to attend to the preamble for the system-identification task rather than encode a single calibration.
3. **Task execution head.** After the preamble the policy executes the actual instruction, conditioning on (instruction, current obs, probe preamble).

The framework treats system identification as an *in-context* task — analogous to how an LLM treats a coding-style demo as in-context, but where the "demo" is the agent's own physical probing.

## Why it matters

- **Targets the real deployment blocker for VLAs.** Frontier robot foundation models work in lab and fail in the world because cameras and arms differ. Fine-tuning per deployment is operationally untenable; ICWM moves the adaptation into the policy's context window.
- **Generalizes the ICL pattern.** Shows that in-context conditioning isn't just for task specification — it's a substrate for system identification, dynamics inference, and arguably tool-affordance discovery. The same template extends to other foundation-model adaptations.
- **Empirically beats standard VLA baselines on novel-viewpoint generalization** in both simulation and real-robot experiments — without parameter updates per deployment.
- **Composes naturally with [on-policy distillation](../post-training/on-policy-distillation.md)** — both use the agent's own rollouts as the supervision signal source rather than external data.

## Gotchas & tricks

- **Probes must be safe.** Calibration probes on a physical robot can damage hardware. Probe action space is bounded and reward-shaped during training, but deployment robustness is non-trivial.
- **Probe length is a hyperparameter.** Too short → underconstrains the system; too long → wastes deployment time. The paper picks short preambles; for harder calibration cases the budget may need to grow.
- **Adaptation surface is narrow.** ICWM handles things that show up in short visual+proprioceptive probes (viewpoint, kinematics). It does *not* identify reward functions, environment goals, or task-specific dynamics that probes can't elicit.
- **Plays badly with KV-cache eviction.** The system-ID information is concentrated in the preamble; aggressive [KV cache compression](../inference/kv-cache-compression.md) that drops the preamble re-introduces the OOD-viewpoint failure mode.
- **Sim-to-real transfer of probe distributions** is the real research question; the paper shows the idea works on Franka-class arms but generalization across morphologies is open.

## Sources

- Paper: *In-Context World Modeling for Robotic Control* — Wang, Shi, Fei, Fu, Ji, Gong, Qiu, 2026 — [arXiv:2606.26025](https://arxiv.org/abs/2606.26025) — Fudan / SII.
- Background: *In-Context Learning* — Brown et al., 2020 (GPT-3) — the few-shot-via-context primitive ICWM repurposes.
- Background: *OpenVLA* and Vision-Language-Action surveys — the model class ICWM augments.
