# LLM-as-Judge
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Using one LLM to score or compare outputs of another (or itself) — for preference labels, RM training, benchmark grading, and online evals. The usual validation question is *"does this judge agree with humans enough to replace one?"* **AltRA-Test** (Omar et al., 2026) extends the classical Alternative Annotator Test (AAT) to the common sparse-overlap setting where different items are labelled by different humans. AltRA-Test certifies ~50% of the evaluated judge configurations as valid in real-world experiments.

**Prereqs:** [README.md](README.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [../post-training/dpo.md](../post-training/dpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

A judge LLM takes one of three shapes:

- **Pointwise scorer.** Takes one output, returns a scalar.
- **Pairwise comparator.** Takes two outputs, returns a preference.
- **Rubric-grader.** Takes an output plus a rubric, returns a structured verdict.

All three are used downstream as reward signals (DPO preference pairs, RM training labels, RLHF rewards), as benchmark graders (MT-Bench, Arena-Hard), and for online quality monitoring.

The *validation problem* — "may this judge replace a human?" — is where AAT-family statistical tests come in.

## How it works

### The judge itself

A judge is just a prompted (or fine-tuned) LLM. The prompt includes the item, any rubric, and a format directive. Common gotchas — position bias, verbosity bias, self-preference bias — are mitigated by: paired evaluation with order swap, chain-of-thought with a final verdict token, and ensembling across judge families.

### Validation

Classical AAT: every item has labels from the same panel of human annotators plus the judge. Resample annotations, swap one human for the judge, check if inter-annotator agreement (IAA) stays non-inferior. Works, but assumes overlapping annotators — rare for large crowdsourced datasets.

**AltRA-Test** (Omar et al., 2026) removes the overlap assumption:

1. **Outer bootstrap.** Resample items.
2. **Inner swap.** For each item, pick a random human annotator to drop and compare the judge against them.
3. **Null distribution.** Compute the resulting IAA delta under exchangeability of human vs judge labels.
4. **Decision.** Accept the judge if its IAA with other humans is non-inferior (one-sided test at a chosen $\alpha$).

Proven consistent, and in simulation outperforms the earlier sparse-overlap baselines. In real experiments, ~50% of (judge, dataset) pairs tested pass.

## Why it matters

- **Gates whether a judge should be let into the training loop.** Running DPO against an unvalidated judge is the fastest route to reward hacking and style drift.
- **Works on real preference datasets.** Most RM/DPO datasets have sparse annotator overlap; AltRA-Test is the first AAT-family test designed for that case.
- **Diversity caveat.** Twice-Told Critiques (Gusev, 2026) finds LLM-driven *homogenisation* of critiques in peer review — a reminder that passing AltRA-Test for *agreement* does not certify *diversity* of a judge's decisions.

## Gotchas & tricks

- **Agreement ≠ correctness.** A judge agreeing with the median human may still agree on systematic errors. Pair AltRA-Test with calibration checks against a gold-label slice.
- **Judge-family correlation.** Two judges from the same base model make correlated errors. Ensemble across families (Claude × GPT × Gemini), not seeds of one model.
- **Position bias.** Pairwise judges score the first option higher unless order is swapped. Always report order-swapped means.
- **Verbosity bias.** Longer outputs win more often absent length-controlled rubrics.
- **Self-preference bias.** A judge grading a generation it also produced tends to score it higher — a known issue for self-training loops.

## Sources

- Paper: *AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators* — Nagham Omar, Rotem Dror, Avigdor Gal, Roi Reichart, Nitay Calderon (Technion / University of Haifa / Google Research), 2026 — primary source for the sparse-overlap AAT.
- Paper: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — Zheng et al., 2023 — earlier judge-validation framework.
- Paper: *Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era* — Gusev, 2026 — diversity failure mode of judge-driven critiques.
