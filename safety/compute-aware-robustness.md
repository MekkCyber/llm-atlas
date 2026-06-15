# Compute-Aware Adversarial Robustness

*Depth — measure attack success rate against attacker FLOPs, not against a fixed query budget.*

**TL;DR:** Adversarial-robustness evaluations of LLMs report **attack success rate (ASR) at fixed query budgets**, implicitly assuming every attack costs the same per query. In reality, attack compute per query varies by orders of magnitude (a one-shot template costs ~10⁻³ of an optimization-based jailbreak). **Compute-aware evaluation** normalizes ASR against cumulative attacker FLOPs as a proxy for adversarial effort, giving a comparable currency across attack strategies and a more realistic picture of which models are robust under economic pressure.

**Prereqs:** [_jailbreaks.md](_jailbreaks.md), [_attacks.md](_attacks.md)
**Related:** [safety-case.md](safety-case.md), [low-resource-language-jailbreak.md](low-resource-language-jailbreak.md), [unusual-format-jailbreak.md](unusual-format-jailbreak.md)

---

## What it is

A reframing of LLM adversarial-robustness evaluation. Instead of reporting

> *"under $N$ queries, attack $A$ jailbreaks model $M$ with success rate $X$"*

report

> *"under $F$ FLOPs of attacker compute, attack $A$ jailbreaks model $M$ with success rate $X$"*

The motivation is that "queries" is not a stable cost unit. A template-based jailbreak is one cheap forward pass at the attacker; a gradient-based or search-augmented jailbreak might burn millions of FLOPs assembling each prompt. ASR at fixed query budgets compares unlike things.

## How it works

### Attack cost accounting

For each attack strategy, the paper accounts FLOPs for:

- Adversarial prefix / suffix optimization (gradient steps over a surrogate model, or black-box search rollouts).
- Auxiliary model calls (rewriting, paraphrasing, encoding attacks).
- The target model's own inference (often dominant for fast attacks).

The resulting metric is cumulative attacker FLOPs to reach a given ASR threshold (or, dually, ASR at a fixed FLOPs budget).

### Re-ranking attacks

When plotted on a FLOPs axis instead of a queries axis, the attack landscape re-ranks:

- **Template / format / language-mismatch attacks** stay cheap → look *more* effective per FLOP.
- **Optimization-based attacks** (GCG-style) become *much* more expensive → their per-FLOP ASR drops sharply.
- **Some attacks that look weak at low query counts** turn out to be strong on a FLOPs budget because they spend their queries efficiently.

### Re-ranking defenses

Defenses that look robust under "attacker capped at 100 queries" may be brittle under "attacker capped at $10^{15}$ FLOPs" — the same total compute as a few hours of large-model training. Compute-aware evaluation surfaces that gap.

## Why it matters

- **Apples-to-apples comparison.** Numbers from different attack papers can finally be put on the same axis.
- **Threat-model honesty.** Attackers don't optimize for queries; they optimize for cost. Compute is the right shared currency.
- **Defense planning.** Safety cases need to know "at what compute does the attacker win?", which fixed-query ASR can't answer.
- **Matches the trajectory of red-team automation.** As attacks get more compute-intensive (optimization, RL-trained attack policies), the FLOPs-axis becomes more informative and the query-axis becomes less.

## Gotchas & tricks

- **FLOPs accounting is approximate.** Different hardware, different attention implementations, different model sizes — all change effective FLOPs. Report assumptions explicitly.
- **Target inference cost dominates for cheap attacks.** When the attack itself is one forward pass, "attacker FLOPs" is mostly the target model's own forward — which is the *defender's* compute, ironically. Separate target vs auxiliary FLOPs in the breakdown.
- **Doesn't replace queries entirely.** Some defenses (rate-limiting, anomaly detection) are most naturally analyzed per query. Use both axes; report both.
- **Need fixed evaluation suites.** ASR curves only mean something if the underlying attack set is held constant across the FLOPs sweep.
- **Calibrate against the strongest known attack.** Compute-aware curves can hide brittleness if the attack set excludes recent strong attacks; treat the curve as an *upper bound* on safety, not a proof of it.

## Sources

- Paper: *Risk Under Pressure: Compute-Aware Evaluation of Adversarial Robustness in Language Models* — Ehghaghi, Ecsedi, Chechik, Raffel, Toronto / Vector / Hugging Face, 2026 — [arXiv:2606.11409](https://arxiv.org/abs/2606.11409).
- Related: [_attacks.md](_attacks.md), [_jailbreaks.md](_jailbreaks.md), [safety-case.md](safety-case.md).
