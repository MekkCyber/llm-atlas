# Population-Level Test-Time Search
*Depth — tournament selection over a population of candidate solutions, with iterative repair, as a test-time scaling method.*

**TL;DR:** Standard test-time scaling samples N candidates and picks the best with a verifier or majority vote. Population-level search instead runs an *evolutionary loop*: maintain a population of candidates, score them with a verifier, generate offspring by repairing the weak ones (informed by the verifier's localization), and select survivors via tournament. Compute is spent where it matters — patching almost-correct candidates — instead of independent restarts. MaxProof (2026) uses this to clear IMO/USAMO gold thresholds.

**Prereqs:** [generative-verifier-rl.md](generative-verifier-rl.md)
**Related:** [mcts.md](mcts.md) · [orm.md](orm.md) · [prm.md](prm.md)

---

## What it is

A test-time inference protocol where compute is allocated across a *population* rather than a fixed batch of independent samples. The population evolves over generations:

- **Generation 0** — sample N candidates from the generator.
- **Each generation** — score all candidates with the verifier; select pairs via tournament; produce offspring by repairing the weaker candidate using the verifier's localization; replace.
- **Termination** — top candidate's verifier score crosses a threshold or generation budget exhausts.

---

## How it works

### Tournament selection

Sample two (or k) candidates, compare their verifier scores, the winner survives. Tournament size controls selection pressure — k=2 keeps diversity, larger k collapses the population to the best candidate fast.

### Repair as mutation

Instead of random perturbations, mutations are *informed* — the verifier localizes the failing step and the repair model produces a patched candidate. This is closer to "directed local search" than evolutionary algorithms — but the population structure prevents premature convergence on a single near-correct candidate that has a fatal flaw the verifier can't fix locally.

### Verifier as fitness

The verifier serves as both reward (for tournament selection) and gradient (for repair). Verifier quality bounds the whole loop's quality.

### Budget allocation

Cost per generation = N (generator forward) + N (verifier forward) + R (repair forward) where R = candidates surviving for mutation. Tune (N, generations, R) to your budget; more generations beats larger N at fixed total compute when repair is high-leverage.

---

## Why it matters

- **Concrete proof of test-time scaling on hard reasoning.** Competition-math gold thresholds were reached not by a bigger model but by spending more inference compute the right way.
- **Bridges single-shot and search.** Pure best-of-N is wasteful when the best candidate is "almost right"; pure MCTS is overkill when the verifier can localize errors directly. Population search splits the difference.
- **Generalizes beyond math.** Any task with a generative verifier and an editable artifact (code, plans, proofs, structured documents) admits the same loop.

---

## Gotchas & tricks

- **Diversity collapse.** If the verifier strongly prefers one style, the population converges to it and loses chance of finding alternative paths. Maintain a diversity term in selection (cluster-aware tournament).
- **Repair vs regenerate.** When a candidate's score is near zero, repair is wasted compute — regenerate from scratch. A simple threshold suffices.
- **Verifier compute dominates.** At scale verifier passes outnumber generator passes by population size. Distill verifiers or accept-then-rerank.
- **Termination criteria matter.** Stopping when the top score crosses a threshold can miss the case where the verifier is over-confident; verify the final candidate with a stronger reranker.

---

## Sources

- Paper: *MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling* — Zhang et al., MiniMax + Fudan, 2026 — [arXiv:2606.13473](https://arxiv.org/abs/2606.13473) — introduces the tournament + repair population loop.
