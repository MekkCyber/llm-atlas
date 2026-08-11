# RailCap
*Depth — decode-time contamination mitigation via runner-up capping on greedy fallback.*

**TL;DR:** Existing benchmark-contamination mitigation methods pre-estimate *which* items the model has memorized, then act on the estimate — so their quality is bounded by the estimator's quality. RailCap judges contamination during generation instead: whenever a sample falls back onto the greedy trajectory, cap the *next* token to the runner-up. Suppression accumulates until the response distribution is dispersed enough. Reported to attain the lowest SA-PPG across multiple contaminated models and benchmarks.

**Prereqs:** [../data/decontamination.md](../data/decontamination.md), [README.md](README.md)
**Related:** [sa-ppg.md](sa-ppg.md), [../data/deduplication.md](../data/deduplication.md)

---

## What it is

Contamination mitigation aims to make a contaminated model's outputs look like an un-contaminated model's — usually by *suppressing memorized* trajectories in favor of *distribution-consistent* alternatives. Pre-estimation methods (identify contaminated items, then apply per-item interventions) inherit the fragility of the estimator. RailCap sidesteps estimation by acting *only at generation time* on a signal available in every decode.

## How it works

The core mechanism runs at decode time:

1. **Sample from the model.** Standard sampling (temperature > 0).
2. **Detect greedy fallback.** At each token, check whether the sampled token equals the greedy (argmax) token. Repeated greedy-fallbacks on the same prompt signal the response has snapped onto a memorized trajectory.
3. **Cap to runner-up.** When fallback is detected, force the next token to the runner-up (second-highest-probability token) instead of the argmax.
4. **Accumulate suppression.** Keep applying the runner-up cap on each subsequent detected fallback. Once the response distribution has dispersed sufficiently (measured by trajectory divergence from the greedy path), release the cap.

There is no pre-estimation phase — RailCap runs uniformly across all items and *judges contamination during generation*. Uncontaminated items rarely trigger sustained greedy fallback, so they see minimal intervention.

## Why it matters

- **Robust to estimation error.** No pre-estimation phase → no compounding error from a bad contamination detector.
- **Uniform across items.** Same decode-time rule for every prompt.
- **Empirically the best mitigation on SA-PPG.** Under the per-question probability-gap metric (which exposes over-suppression that G-AP hides), RailCap achieves the lowest gaps.
- **Composable.** Applies on top of any base model without retraining; can be layered with deduplication of the training set.

## Gotchas & tricks

- **Runner-up isn't always the "clean" continuation.** For very peaked distributions, the runner-up may itself be a memorized fragment.
- **Interacts with sampling temperature.** Higher temperature naturally reduces greedy fallback and thus RailCap's activation rate — tune together.
- **Latency cost.** Requires tracking greedy vs sampled tokens at every step; not free at production scale.
- **Doesn't fix train-time contamination.** RailCap is a serving-side patch — the underlying model still contains the memorized capacity. Pair with deduplication ([../data/deduplication.md](../data/deduplication.md)) for the root cause.

## Sources

- Paper: *Zero Gap Is Not Restoration: Stratified Per-Question Probability Evaluation and Step-wise Mitigation of Benchmark Contamination* — Hou, Jiao, Wang, Li, Zhejiang University, 2026 — arXiv:2608.07341.
