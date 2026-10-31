# Adaptive Task Refresh (A-RSI)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A self-improvement loop that separates two questions prior work conflates: *what experience to generate next* and *when to acquire fresh experience*. "What" is pinned to a static library of model-authored programs that generate multimodal questions with independently-checkable reference answers. "When" is a function of training-log statistics — reward variance and accuracy — that trigger a *task-pool refresh* when the learning signal declines. No external labels, no majority voting, no reliance on validation or test scores.

**Prereqs:** [_rl](_rl.md), [rlvr](rlvr.md)
**Related:** [rl-prompt-curation](rl-prompt-curation.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

Self-improvement loops (R1-Zero style, "generator trains itself on its own output") usually struggle on two separate axes:

- **Supervision quality.** Majority-vote or self-consistency targets depend on the model's own uncertain predictions. Noise compounds across rounds.
- **Data staleness.** Fixed task pools stop producing a learning signal once the model has solved them. Papers often conflate "regenerate tasks" with "find better supervision," but these are distinct.

A-RSI decomposes:

1. **Task-generation contract.** A static, hand-audited library of model-authored *programs*. Each program emits multimodal questions and derives their reference answers mechanically. Chosen properties of the reward and refresh rules are *automatically verified*; the generated answers are *separately audited*.
2. **Refresh trigger.** The training loop monitors reward variation and accuracy on the current task pool. When the signal declines, swap in fresh tasks from the library. No test-score checking in the loop.

## How it works

- **Separate verification axes.** The reward and refresh rules are machine-checkable (do they satisfy stated invariants?). The generated answers are audited independently. Only training-log statistics drive refresh.
- **Reward variation as refresh signal.** When $\mathrm{std}(r)$ or mean accuracy on current tasks plateaus, the signal has saturated — more rollouts here aren't informative. Swap tasks.
- **Fresh tasks on demand.** The program library can emit arbitrarily many; the refresh rule decides how many to introduce and when.
- **Zero external training data.** No validation, no test set, no human labels. The library + checkable answers *is* the data pipeline.

## Why it matters

Multimodal RL has a verifier bottleneck: unlike math, there's no universal grader. The usual workaround is majority-vote or self-consistency, which inherits the model's own uncertainty. A-RSI replaces that with mechanically-derivable answers from a program library, and makes "when to refresh" an explicit knob rather than an implicit epoch-count choice.

Headline numbers: on an 11-benchmark multimodal suite, average rises from **57.77 → 61.05** using 800 fresh tasks. The modularity (what-generation vs when-refresh) is likely to be reused: either half of the method composes with other self-improvement loops.

## Gotchas & tricks

- **Program library is the real bottleneck.** The method is only as diverse as the library's generative capacity. If the programs cover a narrow slice of question types, you'll see the same ceiling as any fixed curriculum.
- **Refresh thresholds matter.** Too eager a refresh wastes compute on fresh tasks before learning; too lazy a refresh wastes compute re-solving stale tasks. Paper's exact thresholds aren't abstract-level.
- **Checkable answers vs human-correct answers.** The verifier checks derivable properties, which is weaker than "a human would mark it correct." Depending on the program design, this can introduce measure drift between A-RSI's "correct" and benchmark "correct."
- **Audit is still required.** The authors emphasize that generated answers must be audited separately from the training loop. Without that audit, errors in the library propagate silently.
- **Composes with GRPO-style trainers.** The refresh trigger is orthogonal to the policy update; drop it into any on-policy trainer.

## Sources

- Paper: *A-RSI: Adaptive Task Refresh for Zero-Data Multimodal Self-Improvement* — Y. Liu et al. (IIE / CAS, CUHK, Würzburg, CMU, MPI-IS, Trento, XJTU), 2026 — arXiv:2610.38568.
