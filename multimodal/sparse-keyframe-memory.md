# Sparse Keyframe Memory (EventVLA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** End-to-end memory module for vision-language-action policies that learns *which* observations will matter later, and stores only those. The Keyframe Evidence Memory (KEM) module predicts future-relevance probabilities from the VLA's own latents and writes only high-probability frames into a sparse evidence buffer. EventVLA (Shanghai AI Lab, 2026) reports +40% average success across 17 simulation and 4 real-world memory-required tasks vs. prior memory-augmented VLAs.

**Prereqs:** [README.md](README.md)
**Related:** [_vla.md](_vla.md) · [../agents/context-as-action.md](../agents/context-as-action.md)

---

## What it is

A vision-language-action policy that has to act over long horizons runs into the same memory problem as a long-context LLM: which past observations are worth keeping? Existing approaches either keep everything (saturated buffer, slow), use a separate memory model (dual-system latency), or apply hand-crafted heuristics (brittle). Sparse keyframe memory keeps neither everything nor nothing — it learns a *prediction head* that says, per step, "this frame will matter later" and stores only those.

## How it works

EventVLA combines two memory components:

| Component | Role |
| --- | --- |
| Foundational visual anchors | Fixed slots for initial-context and short-term-context frames |
| Keyframe Evidence Memory (KEM) | Dynamic slots filled with frames the policy predicts will be future-relevant |

KEM is the novel piece. At each step, a small head reads the VLA's latent embeddings and outputs a scalar future-keyframe probability for the current observation. If the probability exceeds a threshold, the frame's embedding is written into KEM; otherwise it's discarded. KEM has a fixed capacity, so high-probability writes evict the oldest stored keyframe.

End-to-end training: the keyframe head's parameters update against the task reward through the policy's gradient path. Frames whose retention helped downstream success get higher predicted probabilities next time around. No supervised keyframe labels required.

The diagnostic benchmark RoboTwin-MeM is released alongside, with 17 non-Markovian manipulation tasks where transient visual evidence must be retained across many steps.

## Why it matters

- The "predict-your-own-future-relevance" pattern is portable beyond VLAs. Long-context LLMs, code agents, mobile-GUI agents (cf. [../agents/context-as-action.md](../agents/context-as-action.md)) all face the same problem and can adopt the same learned-selection idea.
- Single-system architecture: no separate memory model, no dual-system latency. Critical for real-time control.
- +40% success across many tasks is a large absolute gain in a regime where prior methods plateaued.

## Gotchas & tricks

- The KEM head's signal is weak early in training (the policy can't tell what it'll need later). Some warmup with heuristic keyframes helps bootstrap.
- Fixed KEM capacity is a real constraint; too small loses old context, too large brings back the saturation problem. The paper tunes this per task family.
- The prediction head reads from the VLA's latents, so it inherits the backbone's biases — pathological visual scenes can fool both the policy and its keyframe selector simultaneously.

## Sources

- Paper: *EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies* — Yang, Tu, Yang, Mao, Dong, Chen, Peng, Xiong, Cao, Dai, Zhou, Mu, Wang — Shanghai AI Lab + collaborators, 2026 — [arXiv:2606.20092](https://arxiv.org/abs/2606.20092).
- Benchmark: RoboTwin-MeM (released with the paper).
