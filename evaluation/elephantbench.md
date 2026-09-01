# ElephantBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 1,094-question closed-book knowledge probe designed around **long-tail facts with multiple, genuinely divergent accounts** — not "hallucination", but *epistemic myopia*: models recall one legitimate account and silently omit the others. Questions are built by an auditable graph-based pipeline over a low-exposure web corpus, verified against sources and by human annotators.

**Prereqs:** [README.md](README.md)
**Related:** [../data/decontamination.md](../data/decontamination.md) · [../data/quality-filtering.md](../data/quality-filtering.md) · [mmlu.md](mmlu.md)

---

## What it is

Standard factual QA assumes a single canonical answer, so it can't detect models that *systematically favour one side of a contested long-tail fact*. ElephantBench flips the setup: each question has two (or more) legitimate accounts drawn from real disagreements in the corpus, and the score rewards recovering *both*.

## How it works

Pipeline stages, each auditable and rerunnable:

1. **Corpus retrieval.** Pull related documents from a low-exposure web corpus (chosen to reduce train-time contamination).
2. **Divergence mining.** Build a graph over related documents; identify naturally occurring disagreements about the same entity/event.
3. **Multi-account QA.** Convert each disagreement into a QA record with two (or more) answers, each traceable to its originating documents.
4. **Verification.** Each answer is checked against its source and authoritative public web pages, then reviewed by human annotators.
5. **Scoring.** A model gets full credit only if it surfaces both accounts; partial credit if it recalls one.

## Why it matters

Across **32 models**, even the strongest recovers both accounts on only **52.4%** of questions; on nearly all remaining questions it recalls one account but omits the other. Scaling model size and adding test-time reasoning helps but does *not* eliminate the gap. Corpus analysis shows exposure imbalance strongly predicts which account wins, pinning the failure on training data, not decoding.

## Gotchas & tricks

- Not the same failure mode as "hallucination": the answers the model gives are correct — it just *only gives one*.
- Graph-based construction is contamination-conscious but not immune; models trained on newer crawls of the same corpus can start recognizing distinctive spans.
- Blueprint reusable: the pipeline (retrieve → mine divergence → multi-account QA → verify) turns any long-tail corpus into a source-traceable knowledge probe.

## Sources

- Paper: *Blind Men and the Elephant: Probing the Epistemic Myopia of LLMs under Long-Tail Divergent Knowledge* — Pan et al., Tencent, 2026 — [arxiv](https://arxiv.org/abs/2608.28478)
- Code: https://github.com/Tencent/ElephantBench
