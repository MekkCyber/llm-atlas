# Dynamic Benchmarks
*Depth — evaluations that collect items published after an LLM's knowledge cutoff, and their contamination risks.*

**TL;DR:** As LLMs saturate static benchmarks, evaluators turn to **dynamic benchmarks** that continuously sweep in items (news claims, math contest problems, code challenges) published *after* the model's training cutoff — on the theory that a post-cutoff item can't have been memorized. In practice this defense is partial: 17–29% of "post-cutoff" fact-checking claims are still verifiable from pre-cutoff knowledge, either directly or by synthesizing multiple public facts. Post-cutoff ≠ uncontaminated, and undetected contamination can shift Macro-F1 by 10+ points and flip system rankings.

**Prereqs:** [README](README.md)
**Related:** [../data/decontamination.md](../data/decontamination.md)

---

## What it is

Two design axes:

- **Refresh cadence.** Continuous (rolling weekly ingest), periodic (quarterly benchmark drops), or one-shot post-cutoff snapshots.
- **Source.** Naturally published items (news, code contests, math olympiads) or newly-authored items (human writers with novel questions).

Compared to a static benchmark: same task format, but the *item pool* changes over time and is selected to postdate model cutoffs. Popular examples: LiveCodeBench (rolling code problems), LiveBench (rolling multi-domain), several recent fact-checking benchmarks.

## How it works

```
for each candidate item i:
    if publish_date(i) > model.cutoff and dedupe(i, static_benchmarks):
        include i in eval set
```

The dedupe step is usually shallow (exact match, hash overlap). The publish-date check is treated as sufficient guarantee of no leakage.

## Why it matters

- Removes the most obvious contamination path (exact-string memorization).
- Enables ongoing evaluation of a fixed model against new inputs, useful for tracking drift.
- Provides a defense against benchmark-optimization: labs can't specifically train on next quarter's items.

## Gotchas & tricks

- **Derivability contamination.** A "post-cutoff" claim may be verifiable from purely pre-cutoff knowledge. Claim: "Company X's Q3 2025 earnings beat estimates." If pre-cutoff data contains their pre-announcement guidance and analyst expectations, the model doesn't need the post-cutoff report to answer. Empirically 17–29% of post-cutoff fact-checking claims fall in this bucket.
- **Latent memorization.** Model weights leak knowledge in fuzzy ways — canonicalized phrasings, entity relationships, common-crawl echoes. Exact-match dedupe misses this.
- **Score inflation is silent.** Contamination doesn't obviously fail the eval; it *inflates* scores for models trained on richer pre-cutoff data. Paper reports Macro-F1 inflation up to 11.34 points and observes system-rank flips.
- **Cutoff dates are optimistic.** Post-training on more recent web crawls, RLHF refresh cycles, and fine-tuning data all push the effective cutoff later than the labeled one.
- **Novelty ≠ post-cutoff.** A benchmark that only checks the date lets in items that are trivially derivable. A stronger check requires items describing genuinely new *events* — not just new dates.
- **Reporting protocol.** Report the *fraction of items verifiable from pre-cutoff knowledge* alongside headline scores. Without it, cross-model comparisons over time are apples-to-oranges.

## Sources

- Paper: *Novel Claim or Déjà Vu? Rethinking "Contamination-Free" Dynamic Evaluation for Multimodal Automated Fact-Checking* — He et al., 2026 — [arXiv:2607.23514](https://arxiv.org/abs/2607.23514).
- Related: LiveCodeBench, LiveBench — rolling-item benchmark designs.
