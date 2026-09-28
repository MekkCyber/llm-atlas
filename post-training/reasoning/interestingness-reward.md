# Interestingness Reward (Proof-Length / Statement-Length)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A quantitative signal for **which conjectures are worth proving** in automated theorem discovery. Defines a theorem's *intrinsic interestingness* as `proof length / statement length` — a ratio that correlates strongly with independent measures of downstream utility. Trains a 27B model to predict *conditional proof difficulty* (how hard is proving `S` given premises `P`) more accurately than frontier general-purpose models, then uses the score as an objective to bias conjecture generation toward statements outside the existing library.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [prm.md](prm.md), [orm.md](orm.md)
**Related:** [README.md](README.md), [../rlvr.md](../rlvr.md), [../../evaluation/aime.md](../../evaluation/aime.md), [../../evaluation/math500.md](../../evaluation/math500.md)

---

## What it is

LLMs are increasingly able to prove theorems, but proving is only half of automated math discovery — the other half is **picking things worth proving**. Without a way to rank candidate conjectures, an automated system either recreates known lemmas or wanders into trivialities. Interestingness Reward gives that ranking a concrete, learnable target.

## How it works

Two ingredients:

1. **The interestingness ratio.** For a proven theorem, `I(T) = length(proof) / length(statement)`. Short, potent statements with long proofs score high; long restatements of near-tautologies score low. The paper shows `I(T)` correlates with an *extrinsic* utility measure (how often `T` is later cited or reused).

2. **A learned conditional-difficulty predictor.** For a statement `S` conditioned on a premise set `P`, predict `difficulty(S | P)` — how hard a proof would be given those tools. The paper trains a 27B model on this task and reports higher accuracy than frontier general-purpose models on held-out proofs.

To generate new mathematics: propose candidate statements, score by predicted interestingness, keep the top-k, attempt proofs (with a formal prover in the loop), and iterate. The library grows *outward* — Mathlib overlap drops from 91.9% (naive generation) to 30.6% (with the interestingness objective).

## Why it matters

- Turns a fuzzy problem ("what should we prove?") into a **quantifiable, differentiable signal** amenable to standard RL/curation loops.
- Complements process-reward and outcome-reward models for reasoning: those score *how well you prove*, this scores *whether the target was worth proving in the first place*.
- Sets up self-expanding mathematical libraries — a system that keeps generating novel worthwhile statements without human-supplied targets.

## Gotchas & tricks

- The ratio can be gamed: a statement can be padded to increase `length(proof) / length(statement)` in adversarial ways. The paper's use of a *conditional* difficulty predictor (given premises) helps, but this remains an attack surface.
- Correlation with utility is empirical, not definitional — new domains may not preserve the pattern.
- The 27B difficulty predictor is the paper's real primitive; without it, `I(T)` is only measurable *after* a proof exists, not at conjecture-generation time.

## Sources

- Paper: *Learning to Discover Interesting Mathematics* — Patel, Rammal, Hayat, Munos, Kempe — 2026 — [arXiv:2609.28603](https://arxiv.org/abs/2609.28603).
