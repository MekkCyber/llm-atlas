# Claim-Level Reliability (CLR)
*Depth — reallocate test-time compute from sampling more solutions to refuting decisive claims.*

**TL;DR:** CLR is a training-free test-time scaling framework that condenses each reasoning trace into a compact set of *decision-critical claims* and spends verify-budget trying to **refute** those claims rather than confirm them. It exploits the asymmetry that constructing a valid solution needs a flawless path, while refuting a wrong claim needs only one decisive flaw. Under matched compute, CLR beats pass@1 and self-consistency; on GPT-OSS-20B/CMIMC25 it exceeds pass@1 by +27.15 pp and raises self-consistency accuracy from 77.5% → 82.2% with 37% fewer tokens.

**Prereqs:** [prm](prm.md), [orm](orm.md)
**Related:** [long-cot-rl](long-cot-rl.md), [../../evaluation/aime.md](../../evaluation/aime.md)

---

## What it is

A verification-first alternative to sample-and-vote test-time scaling. Instead of drawing K solutions and taking a majority, CLR draws a smaller number of solutions, extracts the *load-bearing claims* from each, and spends the remaining token budget on targeted refutation attempts against those claims. The reliability score is a nonlinear aggregation over per-claim survival probabilities.

## How it works

For a query `q`, matched compute budget `B` tokens:

1. **Sample** `k` full solution traces `τ_1, …, τ_k` under a fixed sampler.
2. **Claim extraction.** For each `τ_i`, prompt the model to enumerate its decision-critical claims `{c_{i,j}}` — the minimal set of assertions such that the answer follows if all hold.
3. **Refutation search.** For each claim `c_{i,j}`, spend `m` rollouts asking the model to produce a counterexample or a decisive flaw. The claim survives if all `m` attempts fail to refute.
4. **Reliability score.** Aggregate per-trace: `R(τ_i) = f(surviving_claims_i, refuted_claims_i)` with `f` nonlinear (one clean refutation dominates many small confirmations).
5. **Answer.** Pick the trace with the highest `R`.

## Why it matters

- **Signal density.** Whole-trace scoring dilutes: a 500-token trace is 90% routine tokens whose logprobs swamp the few decision-critical ones. Claims isolate the signal.
- **Refutation asymmetry.** Constructing a correct trace is a chain (product of correct steps); refuting a wrong claim is a search (one hit suffices). CLR spends compute where the search is easier.
- **Compute efficiency.** Beats matched-cost self-consistency in most settings tested; the 37%-fewer-tokens result is striking because self-consistency is a hard baseline.

## Gotchas & tricks

- Quality of claim extraction is the ceiling. If the model can't articulate its own load-bearing claims, CLR degenerates toward pass@1.
- Refutation prompts need to explicitly instruct the model to try — models default to agreement on their own outputs otherwise.
- Nonlinear aggregation matters: a linear vote lets many low-confidence confirmations outweigh one high-confidence refutation, defeating the mechanism.
- Not helpful when the answer distribution has a single obvious mode — CLR shines when there are competing high-confidence-but-wrong traces to prune.

## Sources

- Claim-Level Reliability Assessment for Efficient Test-Time Reasoning — Sen Xu et al., 2026 — [arXiv:2608.11994](https://arxiv.org/abs/2608.11994)
