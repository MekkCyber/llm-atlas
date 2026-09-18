# XConf — Experiential Confidence Estimation
*Depth — LLM confidence estimation from an episodic memory of graded past attempts, not from introspection.*

**TL;DR:** Existing confidence estimators (introspection, token-probability aggregation, self-consistency) read only the *current* inference. XConf estimates confidence by consulting a store of the model's **past graded episodes**: (task, reflection, stated confidence, outcome, lesson). Given a new task, it retrieves similar past episodes at similar stated confidence and reads off their historical success rate; a Reflect stage restates confidence conditioned on that record. Format-general (no logit access), one extra generation.

**Prereqs:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)
**Related:** [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md) · [../post-training/_rewards.md](../post-training/_rewards.md)

---

## What it is

A calibration-from-experience estimator. Instead of asking *"how confident does this rollout look?"*, XConf asks *"in past cases like this, how often was the model right when it felt this confident?"* — a frequentist estimator over the model's own track record.

Each episode in the store is a record: task, the model's reflection, its self-stated confidence, the graded outcome (correct/wrong), and a short natural-language lesson written after the grade. The store grows during deployment.

## How it works

Two stages at inference on a new task $x$:

1. **Recall.** Produce an initial answer and stated confidence $c_0$. Retrieve $k$ past episodes similar to $x$ that had stated confidence close to $c_0$. Compute the empirical success rate $\hat p$ on that neighbourhood.
2. **Reflect.** Show the model the retrieved episodes (with outcomes and lessons), have it name its recurring failure modes, then restate its confidence $c_1$ conditioned on that record. Use $c_1$ (optionally combined with $\hat p$) as the final estimate.

No logits, no weight updates, one extra forward pass to produce $c_1$. Similarity is a text embedding lookup over stored task descriptions.

## Why it matters

- **Beats self-consistency at ~1/10 the cost.** Across 9 benchmarks × 4 models × 3 families, XConf matches or beats 10-sample self-consistency AUROC on 23/24 comparisons with much lower ECE.
- **Selective prediction gains at agent scale.** Abstaining on the least-confident 10% of episodes raises delivered success by up to 8.7 points on agent tasks.
- **Turns the deployment log into an alignment asset.** The episodic store is a byproduct of running the model — no separate calibration set needed.

## Gotchas & tricks

- **The store bootstraps slowly.** Early deployment has no episodes; the estimator reduces to the initial $c_0$ until enough graded episodes accumulate.
- **Grades must actually arrive.** For tasks where correctness is unverifiable, XConf has nothing to condition on — same failure mode as any experience-based estimator.
- **Task similarity is the bottleneck.** A poor embedding retrieves irrelevant episodes and calibration degrades; a good one (task-domain embedding) is essential.
- **Complements, doesn't replace, PRM/ORM.** Process-reward models score *within* a rollout; XConf scores *across* rollouts. Combine when both signals are available.

## Sources

- Paper: *Confidence Comes from Experience: Experiential Confidence Estimation from Reasoning to Agents* — Zhang, Zhu, Li, Chen, Kumaran, Collier, University of Cambridge, 2026, arXiv 2609.17708.
