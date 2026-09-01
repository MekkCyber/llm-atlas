# J-Zero
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A three-role self-improvement framework — **Challenger, Solver, Judge** — that co-evolves from *zero training data*. The Challenger proposes tasks the Solver struggles with; the Solver attempts them; the Judge, itself trained in the loop, scores the results. Unlike RLVR (fixed verifier) and RLHF (fixed reward model), the verifier is learned alongside the solver, extending self-improvement into **unverifiable domains** where no external checker exists.

**Prereqs:** [rlvr.md](rlvr.md), [_rl.md](_rl.md), [_rewards.md](_rewards.md)
**Related:** [grpo.md](grpo.md) · [cot-reward-model.md](cot-reward-model.md) · [reasoning/orm.md](reasoning/orm.md) · [reasoning/prm.md](reasoning/prm.md)

---

## What it is

Self-play in verifiable domains (math, code) has a natural verifier: run the tests. In unverifiable ones (writing, open-ended reasoning, judgement), the "verifier" is *what you're trying to learn*. J-Zero handles both cases with the same architecture, avoiding the usual fixed-judge assumption.

## How it works

Three cooperating models updated in a loop:

- **Challenger** — proposes new tasks. Its incentive is to surface tasks the *current* Solver fails or is uncertain on.
- **Solver** — attempts the tasks. Standard RL update against the Judge's scores.
- **Judge** — scores Solver outputs. In verifiable domains, the Judge can be grounded to the checker; in unverifiable ones, it's trained by consistency signals and disagreement between Challenger and Solver.

Each iteration:

1. Challenger emits tasks calibrated to current Solver capability.
2. Solver produces multiple attempts.
3. Judge scores them.
4. All three are updated together: Solver on Judge-derived reward, Judge on separation between good/bad attempts, Challenger on eliciting informative disagreement.

The "Zero" in J-Zero is that the starting corpus is not required — the Challenger-Solver-Judge triple bootstraps the training distribution itself.

## Why it matters

- **+4.2 points** on verifiable-task baselines and **+8.0 points** on unverifiable ones, with monotone improvement over at least **10 iterations**. The unverifiable gain is what's novel: co-evolving the Judge unlocks self-improvement where RLVR is silent.
- Template for pushing R1/o1-style scaling laws into domains that don't have unit tests.
- Sidesteps the "reward-hacking-the-fixed-judge" failure mode by letting the Judge respond to Challenger disagreement.

## Gotchas & tricks

- Instability risk: all three roles updating together can collapse — Judge learns to trivially separate, Solver overfits to Judge, Challenger produces degenerate tasks. Papers typically stagger update frequencies.
- In fully unverifiable domains, at initialization the Judge has no ground truth to anchor on; the paper relies on Challenger–Solver disagreement as the initial signal.
- Not strictly zero data in practice — you still need a base model whose priors define what "reasonable" tasks and answers look like.

## Sources

- Paper: *J-Zero: Unified Challenger–Solver–Judge Co-Evolution from Zero Data* — Chu, Jeon, Yang, KAIST, 2026 — [arxiv](https://arxiv.org/abs/2608.26582)
