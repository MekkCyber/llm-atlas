# Uncertainty-Guided Progressive Contraction
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For reasoning SFT, build small Monte-Carlo trees around every teacher CoT trajectory to measure *per-step* uncertainty of the current student. Keep only trajectories whose uncertainty hasn't collapsed under the student yet; progressively contract the pool as the student learns. Reported to match or beat full-data SFT reasoning accuracy using as little as 0.22% of the standard optimization budget.

**Prereqs:** [post-training/fine-tuning/README](../../post-training/fine-tuning/README.md), [_data-curation](../../data/_data-curation.md)
**Related:** [long-cot-rl](../../post-training/reasoning/long-cot-rl.md), [prm](../../post-training/reasoning/prm.md), [rejection-sampling](../../post-training/rejection-sampling.md)

---

## What it is

A learned, student-aware data selector for reasoning SFT. Instead of using a fixed curated set of CoT trajectories, it ranks trajectories by how much the student still has to learn from them, and shrinks the active training pool as training proceeds.

## How it works

Starting from a teacher CoT dataset:

1. **Build MC reasoning trees.** For each teacher trajectory, side-sample a handful of alternate continuations from the current student at each reasoning step. These form a shallow Monte-Carlo tree rooted at the teacher's prefix.
2. **Score per-step uncertainty.** For each step, use the dispersion of outcomes across sampled branches (entropy over verifier outcomes, or agreement rate) as the step's uncertainty under the student.
3. **Trajectory score.** Aggregate step uncertainties into a trajectory-level "unlearned-ness" score.
4. **Progressive contraction.** Train on the top-$k$ highest-score trajectories. Periodically re-evaluate — trajectories whose uncertainty has dropped (student has learned them) are pruned.
5. **SFT the student** on the shrinking pool.

The contraction schedule concentrates compute on trajectories the student can still gain from, dropping the ones it already handles.

## Why it matters

- **Order of magnitude cheaper.** 0.22% of full-data SFT's optimization budget to match reasoning accuracy, across several benchmarks and model families.
- **Student-aware.** Unlike offline curation (fixed difficulty bins, random subsampling), the selector adapts to the model's current state, so compute always goes to the hardest-still-unlearned traces.
- **Process-level signal.** Uncertainty is measured per reasoning step, so a trajectory with mostly-learned steps and one hard step is still kept until the hard step is mastered.

Suggests most CoT trajectories in standard reasoning datasets are redundant from the student's perspective — a result consistent with the general small-high-quality-data trend in reasoning post-training.

## Gotchas & tricks

- Branch count per step is the main cost dial. Too few branches → noisy uncertainty estimates; too many → the "cheap" story breaks.
- Recomputing uncertainty every step is wasted work; the paper schedules re-evaluation sparsely.
- Contraction can over-prune: if the pool shrinks below a threshold, intermix a small fraction of random trajectories to avoid distribution collapse.
- Verifier dependency — the uncertainty signal relies on being able to score each branch's outcome. For non-verifiable domains, you need a proxy judge.

## Sources

- Paper: *Learning More by Training Less: Uncertainty-Guided Progressive Contraction for Efficient Reasoning Post-Training* — Wang et al., 2026 — [arXiv:2610.12933](https://arxiv.org/abs/2610.12933)
