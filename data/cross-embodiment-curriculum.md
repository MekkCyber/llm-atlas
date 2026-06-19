# Cross-Embodiment Data Curriculum

*Depth — Kairos's pretraining curriculum: open-world videos → human behavioral data → robot interaction data, framed as a developmental scaffold.*

**TL;DR:** A world model intended to drive embodied agents needs to learn *what the world looks like*, *how humans act in it*, and *what robots can do in it* — but these data sources are heterogeneous in scale, label density, and action structure. Kairos organizes them into a **three-stage developmental curriculum**: open-world videos first (largest scale, passive), human behavioral data next (intermediate scale, action-structured), robot interaction data last (smallest scale, full observation–action loops). Each stage scaffolds the next.

**Prereqs:** [_data-curation](_data-curation.md)
**Related:** [quality-filtering](quality-filtering.md)

---

## What it is

A staged pretraining mix for embodied world models — the data-side analogue of curriculum learning on the modeling side. Same model trained sequentially on increasingly action-structured data, rather than a single mixed-distribution corpus.

## How it works

**Stage 1: open-world videos (largest, passive).** Untrimmed internet video. Provides scale and visual diversity — the raw substrate of "what the world looks like and how it evolves." No labels, no action structure, no embodiment metadata. The model learns a general visual-temporal prior at scale.

**Stage 2: human behavioral data (intermediate, action-structured).** Egocentric human-activity video (think Ego4D-style), instructional video, hand-tracking data. Carries implicit action structure — humans do things, and the doing is visible. The model learns the *agentive structure* of the world: an action causes an outcome, and outcomes have visible signatures.

**Stage 3: robot interaction data (smallest, full obs–act loops).** Real robot trajectories with logged observations and actions. Smaller scale by orders of magnitude but the only data with *observed action labels*. The model finally sees the full POMDP-style obs → act → next-obs structure that an embodied policy will need to reason about at inference.

**Why staged rather than mixed.** A single uniform mix lets the easier passive-video objective dominate the gradient (largest dataset, simplest signal), and the small-but-critical robot data washes out. The curriculum forces the model to absorb each level of structure in turn — visual prior → action structure → embodied control — so each later stage builds on a usable initialization rather than learning from scratch.

The framing is **cross-embodiment** because Stage 3 includes data from multiple robot morphologies; the model is expected to share its world-prediction backbone across embodiments while specializing only at the action-decoding head.

## Why it matters

- **Bridges the data-scale mismatch.** Open-world video has 10⁹+ hours; robot interaction has 10³–10⁴ hours. A uniform mix drowns the robot signal; a staged curriculum preserves it.
- **Maps the "developmental pathway" intuition** (children watch → mimic → manipulate) onto a data mixture. Whether or not the intuition holds rigorously, the resulting model performs well on embodied downstream tasks per the Kairos paper.
- **Reusable backbone.** The Stage 1 + 2 backbone is embodiment-agnostic. Different robots only need to specialize Stage 3, sharing the rest. This is a much cheaper deployment story than per-robot pretraining.
- **General template for foundation models for physical AI.** Likely to become the standard data-curriculum shape as more open Physical-AI foundation models emerge.

## Gotchas & tricks

- **Stage boundaries are soft, not hard.** The Kairos abstract describes a "progressive developmental pathway," but exact transition schedules (data fractions, step counts) aren't disclosed. Practitioners likely need to tune blend ratios per checkpoint.
- **Catastrophic forgetting between stages.** Standard mid-training fix: keep a small replay buffer of the prior stage's data mixed into the current stage. Otherwise the model forgets the visual prior while learning robot control.
- **Robot data heterogeneity.** Multiple embodiments in Stage 3 means action-space heterogeneity. Either map all actions to a common low-dim manifold or condition action decoding on embodiment ID; Kairos does the latter implicitly via shared backbone + per-embodiment heads.
- **Stage 1 quality dominates downstream behavior.** Garbage internet video → garbage world prior. Stage 1 quality filtering (deduplication, low-quality removal) matters disproportionately because it's where most parameter-updates land.

## Sources

- Paper: *Kairos: A Native World Model Stack for Physical AI* — Kairos Team, 2026 — [arXiv:2606.16533](https://arxiv.org/abs/2606.16533).
- Related: *Ego4D* — Grauman et al., 2022 — canonical large-scale egocentric human-activity dataset (Stage 2 data shape).
- Related: *Open X-Embodiment* — multiple authors, 2023 — multi-embodiment robot trajectory dataset (Stage 3 data shape).
