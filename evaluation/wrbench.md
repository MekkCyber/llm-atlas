# WRBench
*Depth — benchmark for whether video world models maintain a persistent state core under camera-induced observability changes.*

**TL;DR:** Most current "world models" are graded on perceptual quality of generated frames. WRBench (Lu et al. 2026) argues this misses the point — a *world* model must maintain object/event state through occlusion, off-screen interval, and camera motion, not just produce realistic frames. The benchmark constructs scenes where a dynamic event happens while not directly observable, and tests whether the model's continuation is **state-consistent** rather than merely plausible. Current SOTA video world models fail.

**Prereqs:** *(none)*
**Related:** *(none in graph yet)*

---

## What it is

A diagnostic benchmark for video / world-model systems. Each scenario tests a specific failure mode:

- An object is placed on a table; the camera pans away; an event modifies the object off-screen; the camera pans back. Does the model's continuation reflect the modification?
- A character starts an action; the camera moves; the action progresses off-screen. Does the model produce a frame consistent with the elapsed action time?

Scoring is **state attribution**, not pixel-level reconstruction — the model is graded on whether the inferred latent state of objects/events matches ground truth, not on whether the frames look photorealistic.

## How it works

- **Scene construction.** Synthetic and curated real scenes designed so the answer depends on persistent state, not on what's in the visible frame.
- **State attribution.** Each scenario has a ground-truth state vector (what's where, doing what, in what configuration) measured at the end of the camera trajectory.
- **Model output → state.** The model generates a continuation; the continuation is parsed (or fed to a verifier model) for the implied state, then compared to ground truth.
- **Camera-induced observability changes.** The key manipulation: the same underlying event with different camera paths should yield the same end-state predictions. Discrepancies are state-tracking failures.

## Why it matters

- Repositions the goalposts for world-model claims (Sora-class, Genie-class). "Looks right" ≠ "remembers right".
- Many recent video-generation papers claim world-modeling capability based on perceptual quality alone. WRBench provides the first systematic test that pierces that claim.
- For embodied AI / robotics, **persistent state** is a prerequisite for planning over partial observations. A benchmark that surfaces its absence is upstream of progress on real planning agents.
- The AGI framing the authors invoke is contested, but the engineering test (does the latent state survive occlusion?) is well-defined and broadly useful regardless.

## Gotchas & tricks

- "State" needs to be operationalized cleanly per scenario; ambiguous state attribution muddles scoring. The paper's scenarios are concrete by design.
- A model could pass WRBench by hardcoding scene-graph extraction outside the world-modeling loop. The benchmark measures *behavior*, not *mechanism* — pair with mechanism probes if you care which.
- Synthetic scenes risk being a narrow distribution; the benchmark should be re-checked against real-world video to confirm the failure mode generalizes.
- Sister concept: object permanence in developmental psychology. Adopting that vocabulary clarifies what the benchmark is actually measuring.
- Doesn't compete with FID-class evaluations — those measure realism, WRBench measures consistency. Both axes matter for usable world models.

## Sources

- Paper: *Current World Models Lack a Persistent State Core* — Lu et al., USTC / X-Humanoid / NLPR-CAS / TU Dresden / Peking U., 2026 — arXiv 2606.20545.
- Adjacent literature: object-permanence probes for VLMs; physical-reasoning benchmarks (PHYRE, IntPhys).
