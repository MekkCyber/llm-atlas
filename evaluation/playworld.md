# PlayWorld

*Depth — a benchmark that evaluates interactive video world models by having a multimodal agent player pursue long-horizon objectives instead of executing a scripted action sequence.*

**TL;DR:** Fixed-action evaluation of video world models unfairly penalizes systems whose optimal control differs from the reference sequence — two world models can pursue the same objective with different actions. **PlayWorld** replaces the scripted actor with a multimodal *agent player* that acts on the model to achieve a stated goal ("turn 360° and check if the scene is consistent"), and scores the resulting rollout along **four dimensions**: geometry consistency, interaction fidelity, out-of-sight evolution, and insight evolution — plus basic video-quality and controllability metrics. 171 scenarios, 9 SOTA world models evaluated.

**Prereqs:** (none — read the two lines above)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Video world models are simulators: given a current frame and a prescribed action, they predict future frames. Prior benchmarks compare models on fixed input-action pairs, which forces every model to reach an objective via one particular sequence. That is unfair for two reasons:

1. Two models with equally good behavior can require very different action inputs to reach the same state (different physical priors, different action encodings, different framerates).
2. Naive action-sequence prescription ignores whether the model is *usable* — a good world model should react correctly to whichever actions you throw at it.

PlayWorld reframes evaluation as: **given a goal, let a multimodal agent operate the world model until the goal is (or isn't) achieved**, then score the rollout. The evaluated axis becomes "does this world model support long-horizon interactive objectives," not "does this world model reproduce a specific reference sequence."

## How it works

**The agent player.** A multimodal LLM (the paper uses a strong closed-source VLM) receives (a) the current rendered frame, (b) the natural-language objective, and (c) the model's available action space. It emits an action, waits for the world model's next frame, and iterates. The player is fixed across all evaluated world models so the comparison is apples-to-apples.

**Scenarios.** 171 short objective-carrying scenarios, each specifying an initial frame and a target ("turn to see whether the environment is consistent," "step into water, observe ripples," etc.). Scenarios stress specific dimensions.

**The four core dimensions.**

- **Geometry consistency** — does the scene remain the same when the camera returns to a previously-seen viewpoint?
- **Interaction fidelity** — do interactive elements (water, doors, moving objects) respond correctly to actions?
- **Out-of-sight evolution** — does the world evolve plausibly *outside* the current field of view (e.g. an event that started off-screen)?
- **Insight evolution** — do relationships / states inferred from limited views persist correctly when new evidence arrives?

Two axis metrics for basic sanity: video quality and controllability.

## Why it matters

- **Interactive vs. cinematic separation.** Prior evals conflated "impressive video" with "usable world." PlayWorld splits them.
- **Puts numbers on the failure modes everyone qualitatively suspected.** Across 9 SOTA models, none are reliable on long-horizon interactive objectives; spatial consistency and persistent state evolution are the weakest axes.
- **Shared scoreboard for the current wave of interactive world models.** Directly comparable evaluations of DreamX-Phi-style, Alaya-EVOKE-style, and prior interactive-video systems become possible.
- **Cheap to extend.** Adding a new dimension is one prompt + a scoring rubric; adding a new scenario is one seed frame + one objective string. The benchmark is designed to grow with the field.

## Gotchas & tricks

- **Agent player quality caps the benchmark.** A weak player will fail to elicit long-horizon behavior even from a good world model, muddling attribution.
- **Objective-formulation bias.** The natural-language objective is part of the eval; two phrasings can rank models differently. The paper standardizes phrasings per scenario.
- **Not a replacement for cinematic benchmarks.** A model that scores well on PlayWorld can still generate visually poor frames; keep the video-quality axis alive.
- **Held-out scenarios.** As with any benchmark, scenario leakage is a risk once models are tuned against it. Track held-out splits and contamination.

## Sources

- Paper: *PlayWorld: Benchmarking World Models with Agent Players over Long-Horizon Objectives* — Ding, Chen, Cai, Xu, Wang, Lu, Li, Chen, Gao, Tao, Wan, Zhao, 2026, [arXiv:2608.13552](https://arxiv.org/abs/2608.13552)
