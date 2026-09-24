# Judge Cascade
*Depth — confidence-gated routing from a cheap decision-only judge to a strong generative judge.*

**TL;DR:** LLM-as-a-judge is the dominant scalable evaluation and reward-computation pattern, but frontier judges are expensive at scale. A **judge cascade** routes each item through a cheap decision-only judge first; when the small judge is confident, its verdict stands; when confidence is low, escalate to a strong (generative) judge. JEV-as-a-Judge (2026) shows that this recovers ~99% of the strong judge's accuracy at a fraction of the cost.

**Prereqs:** [README.md](./README.md)
**Related:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md), [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/rubric-rewards.md](../post-training/rubric-rewards.md)

---

## What it is

An evaluation-cost reduction pattern for LLM-judge pipelines. Two tiers:

- **Tier 1 — cheap decision-only judge.** A small (potentially reward-model-style) classifier that outputs a verdict and a confidence, no long generative rationale. Fast, cheap.
- **Tier 2 — strong generative judge.** A large model (frontier LLM) that generates a rationale and a verdict. Slower, expensive, higher-accuracy.

Confidence gate on tier 1 decides who handles each item; low-confidence items escalate.

## How it works

Per JEV-as-a-Judge:

1. **Score every candidate item with the cheap judge.** Get a verdict and an associated confidence (e.g., logprob margin, calibrated softmax, or a learned confidence head).
2. **Accept confident verdicts.** If confidence ≥ threshold $\tau$, the cheap judge's decision is final.
3. **Escalate uncertain items.** Below threshold, forward to the strong generative judge; use its verdict.
4. **Report weighted metric.** Aggregate across both tiers.

JEV specifically: benchmarked against 16 generative and reward-model judges with blinded human adjudication. On preference and evidence-grounded factuality, JEV is within 3 percentage points of the strongest generative comparator at **0.36%** of its per-call cost. Larger accuracy gaps appear on tasks requiring derivation-checking or resisting adversarially-written wrong answers — but those hard cases concentrate in the low-confidence bucket, so the cascade routes them to the strong judge.

Result: at appropriate confidence threshold, 99% of comparator accuracy is retained at large cost savings.

## Why it matters

- **Cost scaling for LLM judges.** Every RL pipeline that uses an LLM judge (rubric rewards, preference RM, factuality checks) pays per-item judge cost. Cascading turns that from $O(N \cdot \text{strong-cost})$ into $O(N \cdot \text{cheap-cost} + k \cdot \text{strong-cost})$ where $k \ll N$.
- **Complements — not competes with — better judges.** The cascade takes any strong judge as the tier-2 model. When frontier judges improve, cascade accuracy improves too, without redesign.
- **Failure-modes concentrate in low confidence.** The paper's cleanest finding: hard cases (derivations, adversarial wrongs) fall disproportionately into the low-confidence bucket, which validates the confidence gate as the right routing signal.

## Gotchas & tricks

- **Confidence calibration is the whole ballgame.** If tier 1's confidence is uncalibrated, either the cascade over-escalates (no cost savings) or under-escalates (accuracy tanks). Cheap-judge confidence should be calibrated against ground truth on a held-out set before the threshold is picked.
- **Threshold sweep matters.** A single global threshold hides task-conditional accuracy differences. Report accuracy-vs-cost curves, not single points.
- **Cheap judge must be trained on judge-like data.** A reward model repurposed as decision-only judge is not the same as one trained specifically for judgment; JEV trains for the decision task directly.
- **Frozen cascade vs adaptive.** A frozen cascade uses fixed threshold and models; adaptive versions can retrain the cheap judge on the strong judge's outputs. Adaptive gets more savings but risks distillation-cascade collapse if the strong judge is imperfect.

## Sources

- Paper: *JEV-as-a-Judge: Accept When Confident, Escalate When Unsure* — Li, Miao, Krishnan, Padman (CMU), 2026 — [arXiv:2609.26550](https://arxiv.org/abs/2609.26550).
