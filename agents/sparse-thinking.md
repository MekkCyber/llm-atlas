# Sparse thinking
*Depth — interleave short bursts of chain-of-thought only at high-stakes decision points so average per-step latency stays bounded.*

**TL;DR:** For real-time agents (games, UI control, robotics), always-on CoT breaks the latency budget. Sparse thinking trains the agent to emit reasoning *only* at a small fraction of steps — the decision points that actually matter — and act immediately on the rest. Introduced in Gamer, where it keeps a VLA responsive in 3D open-world games.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [./README.md](./README.md)

---

## What it is

A training-and-inference pattern for agents that must act on a strict per-step wall-clock budget. The agent emits two kinds of outputs:
- **Act-only steps** — a direct action token, no CoT. Cheap, fast.
- **Think-then-act steps** — a short CoT followed by an action. Expensive, rare.

A gating signal (learned or heuristic) decides which kind to emit. The ratio of think-steps to act-steps is kept sparse — on games, well under 10% of frames.

## How it works

Gamer's recipe stacks three stages:
1. **Continued pretraining** on tens of thousands of hours of keyboard/mouse gameplay to give the VLM an action prior and a sense of when state is ambiguous.
2. **Sparse-thinking SFT** on trajectories annotated with which steps *should* have reasoning (branching moments, novel state, long-horizon choices). The SFT loss masks CoT on act-only steps so the model learns to skip reasoning there.
3. **Multi-turn GRPO** with trajectory-level rewards that compensate for sparse CoT — the model is rewarded for *outcome*, not for emitting CoT at every step, which stops reasoning collapse during RL.

At inference, the decision to think or not is itself a model output — one special token that gates the next reasoning segment.

## Why it matters

Reasoning models bought gains on math/code by always thinking. Agents that act in real time can't afford that cost on every step. Sparse thinking is the natural compromise: keep the reasoning *capability*, but spend it only where it changes the action. The same pattern should transfer to UI agents, computer-use agents, and robotic controllers — any setting with a hard latency floor.

## Gotchas & tricks

- **Decision-point labeling.** The SFT labels for "this is a think-step" dominate the recipe. Report uses heuristics (branching, novelty) plus LLM-judge augmentation.
- **Reasoning collapse during RL.** Without the trajectory-level reward shape, GRPO pushes the model to skip thinking entirely because act-only steps are cheaper — losing capability. Reward design must keep the CoT option alive.
- **Serving.** Think-steps have variable length; cloud/edge inference splits (perception on edge, reasoning on cloud) handle the tail without blocking real-time control.

## Sources

- Paper: *Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent* — 2026 — [arXiv:2611.18397](https://arxiv.org/abs/2611.18397)
