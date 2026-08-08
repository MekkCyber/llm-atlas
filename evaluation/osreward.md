# OSReward
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standardized cross-platform benchmark for reward models that grade **computer-use agent trajectories** (desktop, mobile, web). Unifies task schema, label format, and comparison protocol so per-step correctness models, trajectory verifiers, and LLM judges can be compared apples-to-apples. Analogue of RewardBench for GUI/computer-use agents.

**Prereqs:** [../post-training/_rewards.md](../post-training/_rewards.md).
**Related:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md) · [../agents/README.md](../agents/README.md) · [README.md](./README.md) · [dataspace.md](./dataspace.md)

---

## What it is

Every computer-use RL paper trains its own reward model on its own harness (screenshot + action + outcome → step correctness) and evaluates on its own held-out slice. OSReward defines a shared protocol so that a single RM can be scored on desktop, mobile, and web tasks under one grading scheme.

## How it works

**Trajectory schema.** For each step: `(screenshot, action, outcome_indicator)`. The RM outputs a scalar (or a per-step Bernoulli) indicating whether the action advanced toward task success.

**Ground truth.** Human-labeled step correctness on a diverse pool of trajectories drawn from multiple agents (both successes and failures — the negatives matter). Labels checked with inter-annotator agreement.

**Protocol.** RMs are scored on per-step accuracy, F1, and (for cross-platform generalization) held-out platforms not seen at training time. Reports both aggregate and per-platform numbers so brittleness is visible.

## Why it matters

- **Agent RL is bottlenecked by RM quality** — bad RMs create reward-hacking loops. Without a shared benchmark, published gains are hard to trust or compare.
- **Cross-platform generalization gap** (train on web, test on mobile) is the biggest failure mode surfaced by OSReward — a signal that current RMs pattern-match to platform surface rather than to task semantics.
- **Standardization pressure.** Once RM leaderboards exist, RM design gets systematic — the way RewardBench pushed preference RMs.

## Gotchas & tricks

- **Trajectory diversity is crucial.** A benchmark drawn from one agent family will reward RMs that match that agent's failure modes rather than "correctness."
- **Human labeling of GUI trajectories is expensive**, and inter-annotator agreement is imperfect for ambiguous steps ("did the user progress?"). Expect a ceiling below 100%.
- **LLM judges may leak** — if the judge is the same model family as the tested agent, it can pattern-match to its own reasoning style. Cross-family evaluation is safer.
- **Doesn't cover trajectory-level rewards** directly — those are aggregated from step scores. Trajectory-only RMs (outcome verifiers) fit awkwardly.

## Sources

- Paper: *OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models* — Cheng et al., HKU / XJTU / Nanjing / USTC / NUS / Fudan, 2026 — [arXiv:2607.28609](https://arxiv.org/abs/2607.28609).
