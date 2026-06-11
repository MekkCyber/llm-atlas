# World-model benchmarks

*Depth — evaluating video-based world models on physics, geometry, and interaction control.*

**TL;DR:** Standard video-generation benchmarks (FVD, semantic alignment, short-horizon coherence) reward "good-looking video" but say little about whether a generated future is *usable* as a world model — whether it obeys physical laws, preserves 3D structure, and stays controllable under interventions. WorldOlympiad (2026) is a three-axis triathlon — physical faithfulness, geometric consistency, interaction fidelity — across gaming, robotics, and real-world scenarios. Current SOTA video models excel on one or two axes and collapse on the third.

**Prereqs:** (none)
**Related:** [README](README.md)

---

## What it is

If video models are going to drive downstream agent training (cf. world-model-as-process-reward setups), the eval can't be "does the next frame look good." It has to test whether the model's predicted futures *behave like a world*. WorldOlympiad operationalizes that with three orthogonal axes:

- **Physical faithfulness** — do generated futures obey collision, gravity, deformation, conservation?
- **Geometric consistency** — does 3D structure (camera-coherence, object identity, shape) survive long-horizon generation?
- **Interaction fidelity** — under controlled inputs (agent actions, perturbations), do the futures respond correctly?

A model gets a *triathlon* score combining all three. Scoring high on appearance alone — the comfortable failure mode of every current SOTA — is no longer winning.

## How it works

### Domains

Three scenario families with different ground truths:

1. **Gaming.** Simulated game environments with known physics and accessible state. Easy to score against ground truth.
2. **Robotics.** Real or simulated robot trajectories with known control inputs.
3. **Real-world video.** Naturally-recorded video where physical/geometric laws can be inferred.

The domains stress different axes — gaming is heaviest on interaction, real-world is heaviest on physics.

### Per-axis evaluation

- **Physical faithfulness.** Generate futures, run physics-validation checks (object permanence, momentum, gravity acceleration, collision response). Score by violations.
- **Geometric consistency.** Multi-view consistency, depth-map stability over time, object-id continuity. Score by drift.
- **Interaction fidelity.** Given controlled inputs (an agent action or a counterfactual perturbation), do the generated futures respond correctly? Score by deviation from expected response.

### Triathlon score

Geometric mean or worst-of-three across axes — by design, you can't compensate for a 0 on interaction with a 100 on appearance. Forces models to be *all-around* world models, not just pretty generators.

## Why it matters

- **Right metric for downstream agent training.** Role-Agent-style ([dual-role-self-play](../agents/dual-role-self-play.md)) RL using a world model as the process-reward signal needs the world model to actually be a world model.
- **Exposes the appearance-vs-substance gap.** Most modern video models score well on visual quality; WorldOlympiad shows the gap on interaction and physics is wide and persistent.
- **Encourages 3D-aware architectures.** Models that explicitly reason about 3D structure (NeRF-flavoured, voxel-conditioned, or camera-aware diffusion) tend to dominate the geometric axis.

## Gotchas & tricks

- **Per-axis scoring is brittle.** Each axis depends on auxiliary models / checkers; their errors propagate into the score. Compare *relative* scores carefully.
- **Domain-specific bias.** A model trained on gaming data wins gaming axes for free; one trained on robotics wins robotics. Average across domains, not within one.
- **Doesn't measure long-horizon planning.** WorldOlympiad scores generated futures, not how well an agent plans through the world model. Pair with agent-task evals downstream.
- **Triathlon aggregation is opinionated.** Worst-of-three penalizes specialists harshly; geometric mean is gentler. Report both raw axes alongside any aggregate.

## Sources

- Paper: *WorldOlympiad: Can Your World Model Survive a Triathlon?* — Anonymous (ZJU / DAMO Alibaba / HKUST / Monash), 2026 — [arXiv 2606.11129](https://arxiv.org/abs/2606.11129).
- Background context: world-model-as-process-reward setups, e.g. [dual-role-self-play](../agents/dual-role-self-play.md).
