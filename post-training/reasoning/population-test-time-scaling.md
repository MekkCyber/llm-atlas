# Population-Level Test-Time Scaling

*Depth — TTS that maintains a population of candidate solutions, evolves them via verification and critique-conditioned repair, and selects a final answer via tournament.*

**TL;DR:** Standard test-time scaling spends extra compute by **lengthening** a single rollout (long CoT) or by **independent sampling** (best-of-N). Population-level TTS does both: maintain a *population* of candidate solutions, use the same model as a **verifier** to filter, as a **refiner** to repair, and as a **ranker** to run a tournament among survivors. MaxProof on top of MiniMax-M3 uses this to reach 35/42 IMO 2025 and 36/42 USAMO 2026 — above the human gold-medal threshold.

**Prereqs:** [orm](orm.md), [../grpo](../grpo.md)
**Related:** [generative-verifier](generative-verifier.md), [long-cot-rl](long-cot-rl.md), [mcts](mcts.md), [../rejection-sampling](../rejection-sampling.md), [length-penalty](length-penalty.md)

---

## What it is

A test-time scaling strategy that treats a *set* of candidate solutions as the unit of work, rather than a single rollout. The population is generated, filtered, refined, and ranked using the same underlying model wearing different hats. Compared to:

- **Long CoT**: spends compute extending a single chain. No diversity — one bad branch wastes everything.
- **Best-of-N**: spends compute on independent samples; picks the best by an external verifier. Diversity but no refinement.
- **MCTS / tree search**: maintains a tree of partial solutions, expanding nodes. Strong but expensive and hard to combine with critique.

Population TTS sits between best-of-N and tree search: it keeps a flat population (cheap), but enriches each member via verification + repair (high quality), and uses a tournament rather than a single argmax (robust selection).

## How it works

The MaxProof loop:

1. **Generate** a population of $N$ candidate solutions from the policy. Diversity comes from sampling temperature and possibly from prompt variation.
2. **Verify** each candidate with the generative verifier. Survivors are candidates the verifier blesses (low false-positive rate is critical here — see [generative-verifier](generative-verifier.md)).
3. **Refine** non-survivors with critique-conditioned repair: feed `(problem, broken candidate, verifier critique)` back to the model and ask for a fix. Repaired candidates re-enter verification.
4. **Tournament**: when the population stabilizes, run a pairwise tournament using the same model as a ranker. Winner is returned as the final proof.

All four roles (generator, verifier, refiner, ranker) are the same model — no separate critic or reward checkpoint.

## Why it matters

- **Crosses the gold-medal threshold on IMO 2025 and USAMO 2026** when the policy alone doesn't. The headline result is that TTS infrastructure, not just model capability, gets you the last marginal gains on hard reasoning.
- **Diversity + refinement + selection.** Each component closes a failure mode of the others: independent sampling alone misses near-correct candidates that need a small fix; repair alone needs a seed of *almost*-right answers; tournament selection mitigates verifier overconfidence on individual candidates.
- **Generalizes beyond proofs.** The pattern applies to any task where verification is cheaper than generation and where there's a meaningful critique surface (code, math, structured generation).

## Gotchas & tricks

- **The verifier is the bottleneck.** A high false-positive rate poisons the tournament; a high false-negative rate kills the population. Verifier engineering (see [generative-verifier](generative-verifier.md)) is more than half the project.
- **Diversity matters early, refinement matters late.** Sampling temperature high in stage 1, low in repair.
- **Population size scaling**: more candidates beat more refinement iterations up to a point; beyond that, more iterations on a small set wins. Tune empirically.
- **Cost accounting**: $N$ candidates + verification + refinement passes + a tournament can easily 100× a single rollout. Cost-effectiveness shows up only on problems hard enough that single-rollout success rate is low.

## Sources

- Paper: MaxProof / MiniMax-M3 — Zhang et al. (2026) — [arXiv:2606.13473](https://arxiv.org/abs/2606.13473)
