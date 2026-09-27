# WROP — World Reasoning with Object Permanence
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 150-task benchmark, borrowed from developmental psychology, that probes whether video-generation "world models" encode **object permanence** and **solidity**. Tasks are grouped into six cognitive categories (occlusion, containment, support, continuity, solidity, cohesion). Off-the-shelf video generators fail the majority; a targeted fine-tune closes much of the gap, showing the deficit is a training-signal problem, not architecture.

**Prereqs:** [README.md](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Object permanence — the knowledge that objects continue to exist when out of sight — is a core cognitive prior humans acquire in infancy. WROP treats it as an *evaluation target* for models that predict future video frames.

Each task is a short video prefix followed by an intervention (an object gets occluded, contained, dropped). The model is asked to complete the video. Scoring compares the completion against the physically-correct answer using both:

- **Frame-level metrics.** Whether the hidden object reappears where it should, with correct motion.
- **Consistency metrics.** Whether the completion respects solidity (no interpenetration), gravity where implied, and containment (contained objects move with their container).

Six categories tile the space (as in classical developmental studies):

1. Occlusion — object passes behind a barrier.
2. Containment — object placed inside a container.
3. Support — object on/off a supporting surface.
4. Continuity — object motion continues consistently.
5. Solidity — no two solids occupy the same space.
6. Cohesion — object doesn't split spontaneously.

## How it works

Video generators are prompted with the video prefix and a text description of the scene, then asked to sample 8-24-frame continuations. Automated evaluators check per-category consistency; a small human-verified subset validates the automated grading.

The paper reports two settings:

- **Zero-shot.** Off-the-shelf video generators score poorly across categories — especially occlusion and continuity.
- **After fine-tune on WROP training split.** Same architectures recover much of the gap, indicating the missing signal was in the training data distribution rather than the model's inductive bias.

## Why it matters

- **Grounds the "world model" claim.** Papers claiming video generators are world models can now be tested against a concrete cognitive prior instead of a vibe check.
- **Actionable for training.** The fact that a targeted fine-tune closes most of the gap suggests curated occlusion/containment data can be added to standard video pretraining recipes with clear ROI.
- **Feeds agent safety.** An agent whose planner relies on a video world model that violates object permanence will make unsafe decisions; WROP gives a screening test.

## Gotchas & tricks

- **Frame-level metrics can miss the point** — an occlusion violation is a *categorical* failure, not a per-pixel FID. Report the consistency metrics primarily; frame quality is secondary.
- **Prompt engineering leaks the answer.** If the text prompt says "the ball hidden behind the box rolls out at t=32", any model can pass; use *neutral* scene descriptions.
- **Sampling temperature matters.** At low temperatures the model repeats the prefix; at high temperatures it hallucinates. Sweep and report the best.
- **Six categories are not equally represented in fine-tune data.** Solidity and containment gain more from fine-tuning than continuity, per the paper's ablations.

## Sources

- Paper: *Training Object Permanence in World Models* — Zhang, Yu, Luo, et al. (incl. Yilun Du, Alan Yuille), 2026, [arXiv 2609.28654](https://arxiv.org/abs/2609.28654) — introduces the 150-task WROP suite and demonstrates zero-shot deficits + targeted-fine-tune recovery.
- Related: developmental-psychology paradigms (violation-of-expectation) that the tasks are drawn from.
