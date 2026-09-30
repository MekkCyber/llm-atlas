# Behavioral shadows of post-training
*Depth — measurable changes in a post-trained model's decisions on tasks unrelated to its post-training data.*

**TL;DR:** Post-training a model on task-specific data (SFT, RLHF, RLVR) does not just update behavior on that task — it shifts the model's next-token distribution on *unrelated* prompts in ways that carry structured, distillable information about the update. These "behavioral shadows" turn post-training changes into a leaky side-channel: capability, not just style or trait, can be recovered from single-word teacher outputs on off-target prompts.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../safety/sleeper-agents.md](../safety/sleeper-agents.md), [../post-training/active-taskless-distillation.md](../post-training/active-taskless-distillation.md)

---

## What it is

An empirical phenomenon and a small family of measurement protocols. The claim: for two models that share initialization but differ in a post-training update, their outputs on prompts *unrelated to the post-training task* differ statistically enough to (a) identify which update was applied and (b) transfer capability from the updated model to another via low-bandwidth supervision.

## How it works

- **Measurement.** Sample a batch of prompts from a distribution *disjoint* from the post-training data (typical setup: generic web text or a held-out task the teacher was never trained on). For each prompt, compare the base model's next-token distribution to the post-trained model's. The KL between the two, aggregated over prompts, is the "shadow."
- **Capability probe.** Rather than only measuring divergence, one can train a student on `(prompt, teacher next word)` pairs from these off-target prompts. If the student improves on the *target* skill of the teacher, the shadow carries capability-relevant information — see [active-taskless-distillation](../post-training/active-taskless-distillation.md).
- **Mechanism.** Post-training globally reweights the token distribution — attention keys and value projections change on tokens the update never explicitly targeted. Shadows reflect that global reweighting projected onto the off-target prompt manifold.

## Why it matters

Two consequences:

- **Safety / interpretability.** Capability-leak channels through unrelated outputs are broader than the earlier subliminal-learning literature (which focused on traits and preferences) suggested. A "no capability transfer over this API" claim needs empirical evidence, not an argument by construction.
- **Practical distillation.** Very-low-bandwidth distillation via off-target prompts is a real technique — no target-task examples, no teacher parameters, no logit access, and yet measurable capability transfer.

The measurement also gives a lightweight fingerprint of what a fine-tune actually did: if you can't attribute the shadow to your intended change, the update probably had unintended effects.

## Gotchas & tricks

- **Distribution matters.** "Unrelated prompts" is not a well-defined set; some off-target distributions carry more capability shadow than others. Use several.
- **Base-model comparison is essential.** A shadow is relative to a reference; without the pre-update model, you can only measure student-teacher agreement, which conflates data effects with post-training effects.
- **Denoising helps.** Because the shadow is small, averaging over many prompts and many samples reduces noise; single-sample comparisons are dominated by decoding stochasticity.
- Shadows are asymmetric: a fine-tune that adds capability leaves a stronger shadow than one that only shifts style or refusal behavior.

## Sources

- Paper: Post-Training Leaves Behavioral Shadows on Unrelated Decisions — Zhang, Jing, Zeng, Li, Wang, Gong, 2026 — [arXiv:2609.29233](https://arxiv.org/abs/2609.29233)
- Prior line: subliminal learning of traits and preferences via unrelated generations.
