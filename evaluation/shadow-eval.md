# Shadow Evaluations

*Depth — an eval methodology for open-ended AI research capability: agents attempt an unpublished paper's central research question, and the paper's actual authors grade the output.*

**TL;DR:** Existing measures of "can agents do AI research" are stuck between narrow verifiable benchmarks (exclude the interesting parts) and blind peer review (overstretched, noisy). Shadow evaluations plug the gap: give the agent the central open-ended question from a real, high-quality *unpublished* paper; let it work with real budget (six days, thousands of dollars of compute); then have the paper's original authors grade the result. The authors know what a good answer looks like and are willing to declare rejection.

**Prereqs:** *(none — a novel evaluation methodology)*
**Related:** [README.md](./README.md), [../agents/README.md](../agents/README.md)

---

## What it is

A capability-eval design specifically for **open-ended AI research automation**. The three defining constraints:

- **Unpublished, high-quality source paper.** Prevents both training-set leakage and cherry-picking to already-solved questions. NeurIPS submissions during review windows work well.
- **Author-graded.** The paper's actual authors — not third-party reviewers — grade the agent's output. They have the deepest possible ground truth about what a successful answer looks like.
- **Realistic budget.** Days, not minutes; thousands of dollars of compute, not toy inference. Matches the resource envelope of a real ML researcher on a real project.

The output is a *qualitative* judgment (accept, reject, plus reasoned commentary), not a leaderboard score.

## How it works

1. **Recruit unpublished papers.** Author teams volunteer their in-review submissions.
2. **Extract the central research question.** Distill the paper's contribution into a one-sentence problem statement.
3. **Give the agent budget.** Six days of wall-clock, thousands of dollars of compute — matched to the scale a researcher would use.
4. **Author grading.** Original authors read the agent's output and its trajectory logs, decide accept/reject, and document failure modes.
5. **Robustness check.** Re-run with a different frontier model + scaffold to test whether failures are model- or scaffold-specific.

In the reported runs, both papers were **unambiguously rejected** by their own authors.

## Why it matters

- **Ground truth from the source.** Peer review's noise mostly comes from reviewer variability; authors know the answer.
- **No leakage.** Unpublished papers weren't in any model's training set at run time.
- **Diagnoses *what* fails.** The paper distills the failure modes into five recurring categories: judgment about the publishable bar, uncreative response to design shortcomings, ineffective backtracking, poor resource awareness, and instruction drift.
- **Reusable methodology.** The shape (private ground-truth author + real budget + qualitative rubric) transfers to other open-ended domains: engineering, product design, editorial, medical diagnosis with expert clinicians as author-analogues.

## Gotchas & tricks

- **Sample size is intrinsically tiny.** Two papers isn't a benchmark; it's an existence proof of failure. Sales-cycle to recruit authors + wait for their review outcomes caps throughput. Read shadow-eval results as case studies, not summary statistics.
- **Author bias.** Original authors are simultaneously the most-informed graders and the most-invested; they may grade harshly on their own paper's question. The paper mitigates with structured rubrics and expert-reviewer surveys, but the bias exists.
- **Compute + time budget is expensive.** Real per-eval cost is thousands of dollars and days of wall-clock. Scaling to N=50 papers is a significant program, not a weekend project.
- **Cannot test *creative* research questions the author didn't write.** The eval measures whether the agent can answer *the human's* research question — not whether it could have found a better question.
- **Release matters.** The paper releases expert reviews, survey responses, agent repos, and logs — critical for community follow-up and cross-lab comparison.

## Sources

- Paper: *Can AI agents conduct open-ended AI research? Early evidence from two case studies* — Kirgis, Kapoor, Schwartz et al., 2026 — introduces shadow evaluations and reports the first two cases. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
