# Population-Level Test-Time Scaling

*Depth — a test-time loop where a single model plays generator, verifier, refiner, and ranker over a population of candidate solutions.*

**TL;DR:** Standard test-time scaling is "sample $N$, pick the best by some scorer." Population-level test-time scaling treats the candidate set as a **population**: the same model **generates** new candidates, **verifies** them, **refines** failures via critique-conditioned repair, and **ranks** survivors via tournament selection. A single released model fills all four roles. With a low-false-positive [generative verifier](generative-verifier.md), this loop drives MiniMax-M3 to **IMO 2025 35/42** and **USAMO 2026 36/42** — beating the human gold-medal threshold on both.

**Prereqs:** [generative-verifier.md](generative-verifier.md), [orm.md](orm.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [../rlvr.md](../rlvr.md), [length-penalty.md](length-penalty.md), [../../evaluation/aime.md](../../evaluation/aime.md)

---

## What it is

A test-time search loop that uses a single multi-capability model in four roles:

| Role | Action | Inputs | Output |
| --- | --- | --- | --- |
| **Generator** | Sample candidate proof / solution | Problem | Candidate $o_i$ |
| **Verifier** | Run a generative-verifier audit | Problem, candidate | Accept / reject (+ critique) |
| **Refiner** | Critique-conditioned repair | Candidate, critique | Refined candidate |
| **Ranker** | Tournament selection among survivors | Pair of candidates | Winner |

Where best-of-$N$ is a flat scoring pipeline (sample $N$, score, top-1), population scaling is an *iterated* loop — the population is replenished by refinement, pruned by verification, and reduced to one by tournament ranking.

## How it works

```
P ← { generate(problem)  for i in 1..N₀ }            # seed population
repeat:
    for o in P:
        v, critique ← verifier(problem, o)
        if not v:
            o' ← refine(problem, o, critique)
            replace o with o' in P
    drop o ∈ P that fail verification after refinement
until budget exhausted or |P| ≤ N_final
return tournament(P)                                   # pairwise ranker
```

Three structural commitments make this work:

1. **One model, four roles.** Generator and verifier are the same LM in different prompt modes. Otherwise critique-conditioned repair would be incoherent.
2. **Low-FPR verifier.** If the verifier accepts wrong proofs, the population gets polluted and tournament selection can pick a wrong winner. The whole pipeline rides on verifier calibration ([generative-verifier.md](generative-verifier.md)).
3. **Tournament selection over scalar scoring.** Pairwise comparisons via the model itself are more robust than calibrating a global scalar reward across heterogeneous proofs.

## Why it matters

- **Frontier-math headline.** Pushes a 100B-class model past human gold-medal thresholds on IMO and USAMO with a *single* released checkpoint plus a test-time loop.
- **Decouples capability from inference cost.** Population scaling is a knob: more compute → more candidates → higher pass rate, on the same model.
- **The right framing for problems where verification is asymmetric.** Math proofs, code with tests, formal-system theorems — anywhere "check is easier than write" — fits this pattern.

## Gotchas & tricks

- **Verifier FPR is the load-bearing parameter.** If it's not tiny, the population concentrates around false positives.
- **Refinement budget matters.** Without enough refine→re-verify rounds, the loop terminates with a sparse population of low-quality candidates.
- **Tournament ranker is not a scalar reward.** Pairwise selection avoids global-calibration drift that scalar scoring (and even some generative verifiers' confidence outputs) suffer from.
- **Length & format collapse.** Encouraging the generator toward longer, structured proofs is necessary so the verifier has structure to audit ([length-penalty.md](length-penalty.md) interacts).
- **Compute scaling curves.** Pass rate vs compute plateaus when verifier FPR floors are hit, not when population is exhausted — the right axis for ablation is verifier quality, not $N$.

## Sources

- Paper: *MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling* — Zhang et al., MiniMax, 2026 — [arXiv:2606.13473](https://arxiv.org/abs/2606.13473).
- Related: [generative-verifier.md](generative-verifier.md), [orm.md](orm.md), [long-cot-rl.md](long-cot-rl.md).
