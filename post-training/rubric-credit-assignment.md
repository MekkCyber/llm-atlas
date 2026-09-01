# Rubric-to-Code Credit Assignment (RCCA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An RL recipe for outputs that satisfy **many independently-scorable requirements** (e.g. an interactive web app with per-feature rubrics). Standard GRPO collapses these into a single sequence-level reward and applies the same advantage to every token. RCCA instead uses **rubric-level functional feedback** and an evaluator that attributes each rubric outcome to specific code spans, producing a **localized** advantage over the responsible tokens.

**Prereqs:** [grpo.md](grpo.md), [_rewards.md](_rewards.md)
**Related:** [cot-reward-model.md](cot-reward-model.md) · [reasoning/prm.md](reasoning/prm.md) · [rlvr.md](rlvr.md)

---

## What it is

For structured outputs like generated web apps, the reward is really the AND / weighted-sum of many local requirements (a working button here, correct routing there, a specific CSS behaviour). A single trajectory reward washes those apart. RCCA turns the reward into a **rubric-attributed** signal that GRPO-style policy gradients can actually use.

## How it works

Three components:

1. **Rubric-structured tasks.** Each task ships with an explicit list of functional rubrics — one per user-facing requirement. Training data is built around these rubrics, not around free-form specs.
2. **Hierarchical reward.** Failures are categorized into **format / source-code / runtime / functional** buckets and rewarded separately, so an app that runs but fails one feature scores above one that doesn't compile. This gives smooth gradient even when many rubrics fail.
3. **Attribution to code spans.** An evaluator inspects the generated code and links each rubric outcome (pass/fail) to the specific code spans responsible (event handlers, state updates, DOM fragments, CSS selectors). Those spans' tokens receive the corresponding advantage; unaffected tokens receive none.

The RL update is otherwise GRPO — group of samples, ratio-clipped PPO objective, KL to reference — but with per-span advantages instead of one shared per-response scalar.

## Why it matters

- **Ling-RCCA-Flash** reaches **41.25 on MiniAppBench**, **+32.20** over Ling-3.0-Flash, slightly above Claude Opus 4.5.
- **76.19 on ArtifactsBench**, **+4.48** over the SFT model, setting a new leaderboard top under the official setting and beating GPT-5 by 3.64.
- Transfers beyond web apps: any long output split across many independently scorable regions (RPC handlers, DB migrations, structured reports) fits the same template.

## Gotchas & tricks

- Attribution quality is the whole game. If the evaluator can't cleanly link a rubric to code, its advantage smears back to a global signal (and you're back to GRPO).
- Format / source-code / runtime / functional hierarchy is not decorative — it stops the gradient from being 0 whenever runtime fails, which is common early in training.
- Composes with hierarchical / process reward ideas: attribution is process-reward-per-region rather than per-step; the policy update itself is unchanged.

## Sources

- Paper: *Rubric-to-Code Credit Assignment for Reinforcement Learning* — Jin et al., Ant Group, 2026 — [arxiv](https://arxiv.org/abs/2608.27906)
