# Capability-Preserving SFT (CP-SFT)
*Depth — SFT routed through a domain-specific system prompt to minimize interference with base capabilities.*

**TL;DR:** Introduced by nanoAstra (2026) to specialize a VLM for robotic manipulation without destroying general VQA ability. Robot-specific supervision is conditioned on an explicit **robot-only system prompt**, so SFT updates concentrate adaptation on robot contexts while leaving the unconditioned distribution close to the base model. The result: robot specialization + preserved general capability on a single checkpoint.

**Prereqs:** [README](README.md)
**Related:** [multimodal/README](../../multimodal/README.md)

---

## What it is

A conditioning convention for SFT on domain data. Every domain example is wrapped with a system prompt that marks it as domain-specific (e.g., "You are controlling a robot. Produce semantic actions in the following format…"). Non-domain evaluations don't use that prompt, so the model's behavior on general queries stays close to the base distribution.

## How it works

Each training example looks like:

```
<system>: robot-specific prompt (identity, action schema, constraints)
<user>: observation + instruction
<assistant>: semantic action steps
```

The gradient update still adjusts all weights, but the model learns to **gate** its specialized behavior on the system prompt. Empirically this routes adaptation through the prompt-conditioning pathway rather than overwriting the base distribution. At serving time, calling the model without the system prompt recovers near-base general behavior; calling with it triggers the robot policy.

nanoAstra trains on >5,000 hours of robot trajectories mixed with general VQA, using CP-SFT plus a semantic action representation (reusable parameterized primitives with explicit progress descriptions).

## Why it matters

Standard SFT on narrow domain data is a well-known specialization-vs-generality tax. CP-SFT is a cheap, no-architecture-change way to pay less of that tax: one model, two serving modes, and the preservation of general capabilities is empirically visible on VQA while the robot benchmark still lifts (17.8% zero-shot on RoboTwin 2.0 for nanoAstra-9B).

Conceptually related to prompt tuning and to LoRA adapters, but the mechanism here is purely behavioral — no extra parameters, no adapter switching — which makes it compatible with any inference stack.

## Gotchas & tricks

- **The system prompt must be distinctive.** Generic prompts leak into unprompted inference. nanoAstra uses a long, structured prompt that is unlikely to appear naturally.
- **Data mixture still matters.** Mixing general VQA data during CP-SFT keeps unconditioned behavior anchored; pure domain data + system-prompt routing still drifts.
- **Negative transfer is not eliminated, only attenuated.** Hard failure modes (e.g., the model emitting robot-style outputs on non-robot queries) still appear at low rates.
- Scales to action-chunk streaming (inference overlapping with motion) because the prompt framing already structures outputs as discrete decisions.

## Sources

- Paper: A Glimpse of Training VLMs as Foundation Models for Robotic Manipulation — Zhang et al., 2026 — [arXiv:2611.29338](https://arxiv.org/abs/2611.29338)
