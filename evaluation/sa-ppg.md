# SA-PPG (Stratified Aggregate of Per-question Probability Gaps)
*Depth — a per-question, probability-space metric for evaluating benchmark-contamination mitigation.*

**TL;DR:** The dominant metric for judging whether a contamination-mitigation intervention has actually *restored* a model's un-contaminated performance is G-AP (Gap of Aggregate Performance). G-AP averages before differencing, so per-question over- and under-suppression cancel; it also weights questions uniformly, inviting strategies that push solve probabilities onto the clean model's high-frequency hotspots. SA-PPG replaces it: estimate each question's solve probability by sampling, difference against the clean model *per question*, then aggregate within strata defined by the clean model's solve probability.

**Prereqs:** [../data/decontamination.md](../data/decontamination.md), [README.md](README.md)
**Related:** [railcap.md](railcap.md), [../data/deduplication.md](../data/deduplication.md)

---

## What it is

When a model has memorized benchmark items, mitigation methods try to *suppress* the memorized outputs and *restore* what the model would have scored without contamination. Evaluating whether that restoration is genuine — versus a coincidence of aggregate scores — requires a metric that operates per question and in probability space.

## How it works

1. **Estimate per-question solve probability by sampling.** For each question $q$, sample $K$ responses from the model (contaminated + mitigated), estimate the empirical solve probability $\hat p(q)$.
2. **Reference against a clean model.** Get the clean-model solve probability $p_{\text{clean}}(q)$ (from an uncontaminated checkpoint or reference).
3. **Per-question probability gap.** Compute $\Delta(q) = \hat p(q) - p_{\text{clean}}(q)$ — this is signed (over- vs under-suppression).
4. **Stratify by clean-model difficulty.** Group questions into buckets defined by $p_{\text{clean}}(q)$ (e.g. easy: $p \in [0.9, 1.0]$, medium: $[0.5, 0.9]$, hard: $[0, 0.5]$).
5. **Aggregate within strata.** Report the mean absolute (or signed) $\Delta(q)$ within each stratum. Overall SA-PPG is a weighted sum over strata.

The key design choices: **difference before aggregate** (so cancellation is impossible), **probability-space** (so 0/1 correctness cannot hide subtle drift), and **stratification** (so gaming the mean via easy-question tricks is exposed).

## Why it matters

- **Restoration claims get much stricter.** The paper shows prior mitigation strategies' apparent restoration is substantially **overestimated** under SA-PPG — meaning most published "we can restore contaminated models" results should be re-checked.
- **Kills a class of gaming.** Uniform-weighting metrics reward interventions that push probabilities onto the clean model's high-frequency values; stratification makes that visible.
- **Better than a single number.** Reporting per-stratum SA-PPG makes clear *where* an intervention over- or under-suppresses.

## Gotchas & tricks

- **Sampling budget matters.** $\hat p(q)$ needs enough samples per question to be stable; the paper uses $K$ ~ tens.
- **Clean-model choice is definitional.** SA-PPG is a *comparative* metric against a reference. A weaker clean model makes any mitigation look better.
- **Stratum boundaries are hyperparameters.** Boundary choices can shift headline numbers; report SA-PPG per stratum, not just an aggregate.
- **Doesn't distinguish mitigation from ability loss.** Both over-suppression and pure capability loss show up the same way — reason to pair with capability-preservation evals.

## Sources

- Paper: *Zero Gap Is Not Restoration: Stratified Per-Question Probability Evaluation and Step-wise Mitigation of Benchmark Contamination* — Hou, Jiao, Wang, Li, Zhejiang University, 2026 — arXiv:2608.07341.
