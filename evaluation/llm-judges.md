# LLM Judges
*Depth — using a large language model as an evaluator or scorer.*

**TL;DR:** An **LLM judge** is a model prompted (sometimes fine-tuned) to score another model's output — for correctness, helpfulness, preference between two candidates, or arbitrary rubrics. Cheaper and more scalable than human review, judges have become the default evaluation tool for open-ended tasks (RLAIF labels, Arena-style elo, benchmark grading). They are also *attackable*: judges can be swayed by rewording that preserves meaning, order of candidates, or stylistic cues — a problem that the recent **SciCore / RobustReview** work (Wang et al., 2026) measures head-on.

**Prereqs:** [README.md](./README.md)
**Related:** [../post-training/_rewards.md](../post-training/_rewards.md) · [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)

---

## What it is

Three common judge roles:

| Role | Input | Output |
| --- | --- | --- |
| **Pointwise scorer** | (prompt, response) | Scalar quality score or Likert rating |
| **Pairwise judge** | (prompt, response A, response B) | A > B / B > A / tie |
| **Rubric reviewer** | (prompt, response, rubric clauses) | Pass/fail per clause |

Judges can be prompted (zero-shot, with a system prompt) or trained (SFT or preference fine-tuned on gold judgments). Chain-of-thought judges emit a reasoning trace before the verdict (see [cot-reward-model](../post-training/cot-reward-model.md) for the trained variant).

## How it works

Pointwise prompt skeleton:

```
You are judging answers to <task>. Score 1-5 on <axis>. Return JSON {"score": N, "reason": "..."}.
Question: <q>
Answer: <a>
```

Pairwise prompt:

```
Judge which answer is better. Return JSON {"winner": "A"|"B"|"tie", "reason": "..."}.
```

For rubric review (SciCore and other scientific-review setups), the rubric is a list of clauses; the judge is asked to grade each.

**Two-branch judging** (SciCore): run two judges in parallel — one on the full prose manuscript, one on a *structured science core* extracted from the manuscript — and average. The prose branch catches alignment-relevant features; the core branch is rhetoric-stripped and robust to rewrites.

## Why it matters

- **Scales eval.** Human review doesn't scale to the millions-of-outputs regime of RLAIF, Arena, or iterative benchmarking.
- **Open-ended tasks.** Rule verifiers can't score "is this response helpful?"; judges can.
- **Reward in disguise.** A pairwise judge *is* a preference reward model; the DPO loss uses one implicitly.
- **Trust requires measurement.** As judges enter decision loops (hiring, grading, review), their robustness-vs-discrimination tradeoff needs its own metrics — this is what *rhetorical robustness* names.

## Gotchas & tricks

- **Positional bias.** Pairwise judges prefer the first candidate; randomize order and average.
- **Length bias.** Longer answers are over-rated; normalize or train against a length-matched baseline.
- **Rhetorical gameability.** Content-preserving rewrites can swing judgments. The SciCore paper shows this is widespread and introduces *RobustReview* (1,260 manuscripts × 30 configs) to quantify it.
- **Robustness vs discrimination.** A judge that doesn't move with rewrites often *also* stops distinguishing genuinely different responses — collapsed scoring. Treat the two metrics as a pair.
- **Judge-model drift.** Switching the judge from GPT-4 to Claude to a smaller open model changes the rankings; publish the judge version alongside results.
- **Self-preference.** A judge from family X tends to prefer answers from family X; use a *different* family as the judge when possible, or use an ensemble.
- **Human alignment ≠ robustness.** The paper finds the two rankings diverge; measure both before using a judge in loops where humans will later audit.

## Sources

- Paper: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — Zheng et al., 2023 — the canonical LLM-judge-for-evaluation study.
- Paper: *A Missing Piece for Trustworthy AI Reviewers: From Benchmarking Rhetorical Robustness to SciCore Review* — Wang et al., 2026 — [arXiv:2609.39027](https://arxiv.org/abs/2609.39027). Introduces rhetorical robustness, the RobustReview benchmark, and the SciCore two-branch judge.
- Paper: *Constitutional AI / RLAIF* — Bai et al., 2022 — judges as the preference-label source for RL post-training.
