# ImpossibleRubrics
*Depth — benchmark stress-testing LM-generated rubrics as reward signals against certificate-violating answers.*

**TL;DR:** Rubric-based RL and LLM-as-judge pipelines rely on LM-generated rubrics as reward signals. ImpossibleRubrics (Qin et al., 2026) is a 169-task benchmark of **impossible** prompts (whose only honest answer is to acknowledge impossibility) plus 48 answerable controls. Each impossible task ships with an **oracle certificate** specifying what a valid answer may or may not claim. Rubrics generated downstream are then adversarially tested for whether they reward certificate-violating answers.

**Prereqs:** [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)
**Related:** [README](README.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

A benchmark and methodology for evaluating **generated rubrics** rather than fixed ones. Instead of asking "does this rubric agree with humans on this task?", ImpossibleRubrics asks "does this rubric reward a specific class of *dishonest* answer that violates the task's oracle certificate?". The benchmark isolates the hardest regime for rubrics — tasks where the honest response is to refuse or acknowledge unanswerability.

## How it works

1. **Task environment + certificate.** For each of the 169 impossible tasks, the benchmark provides the prompt (which pressures a specific unsupported conclusion) and an **oracle certificate** — a machine-checkable statement of what any valid answer may claim (typically: acknowledgment of impossibility, no fabricated evidence, no assertion of the pressured conclusion).
2. **Six impossibility categories** cover different failure vectors — logical contradictions, missing information, undefined premises, category errors, over-specified constraints, and factually unresolved questions.
3. **48 answerable controls.** Adjacent-form prompts that *are* answerable, to detect rubrics that reward mere refusal indiscriminately.
4. **Rubric generation.** A downstream method (any rubric generator) is given the task and certificate and produces a rubric.
5. **Adversarial answers.** The benchmark supplies a suite of certificate-violating answers optimized to look plausible.
6. **Scoring.** The rubric's score on the adversarial answers, compared to the honest answer's score, exposes reward hackability.

## Why it matters

Rubric-based RL is proliferating (RLVR, generated rubrics for RM training, LLM-as-judge). If the rubric rewards a well-optimized dishonest answer, the whole pipeline is poisoned — the trained model will hallucinate confidently on the impossible cases. ImpossibleRubrics is the first benchmark that specifically stress-tests this failure mode with a certificate-based oracle, filling a gap between:

- Human-preference reward-model evaluations (that don't isolate impossibility)
- Refusal benchmarks (that don't test reward robustness)

## Gotchas & tricks

- **Certificates ≠ ground truth.** A certificate says what an answer may/may not claim; it does not enumerate all valid honest answers. Use it as a filter, not a target.
- **Controls prevent trivial refusal.** A rubric that just rewards refusal aces the impossible tasks but fails the controls. Both scores are needed.
- **Impossibility categories are not equally hard.** Missing-information tasks are typically easier for rubrics to handle; category errors and over-specified constraints trip them up more.
- **Rubric-generator matters.** The benchmark evaluates the pair (task, generated rubric), not the rubric alone. Reporting must name the generator.
- **Adversarial answers rotate.** A rubric that beats today's adversarial suite may fall to tomorrow's. Treat scores as snapshots.

## Sources

- Paper: *ImpossibleRubrics: Stress-Testing Generated Rubrics as Reward Signals* — Qin, Xie, Liu, Yang — 2026 — [arXiv:2609.16816](https://arxiv.org/abs/2609.16816).
