# Theorem Interestingness — Proof/Statement Length Ratio
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A metric for *which theorems are worth proving*, defined as **length(proof) / length(statement)**. This ratio correlates with an extrinsic measure of downstream utility (how much a theorem is reused inside Mathlib), and optimising for it lets a conjecture-generation model produce theorems substantially outside its training distribution while remaining formally verifiable.

**Prereqs:** [README.md](README.md)
**Related:** [math500.md](math500.md), [aime.md](aime.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Once LLMs can *prove* theorems, the bottleneck shifts to *choosing* which theorems to prove. Volume without selection floods a library with trivial restatements. The paper proposes an **intrinsic interestingness** score defined purely in terms of the theorem itself:

$$\text{Interestingness}(T) = \frac{\ell(\text{proof}(T))}{\ell(\text{statement}(T))}$$

A short statement whose proof is long is *dense information* — it condenses hard work into a memorable claim. Both lengths are measured in tokens of the target formal system (Lean, Isabelle); other length proxies work but are noisier.

## How it works

The metric composes from a primitive: a **proof-difficulty predictor** — a 27B model trained to estimate proof length in Mathlib conditioned on the statement and a set of available premises. Given this predictor, you can:

1. **Rank generated conjectures** by predicted proof/statement ratio.
2. **Guide proof search**: allocate more search budget to conjectures whose difficulty is high but not intractable.
3. **Optimise a generator** with the score as a reward signal — the paper trains a conjecture-generation model to maximise the score under a verifiability constraint.

Optimising the score reduces Mathlib-overlap of generated statements from **91.9% → 30.6%**, i.e. produces theorems substantially outside the training distribution while keeping them formally verifiable. The 27B difficulty predictor beats frontier general models on proof-difficulty ranking.

## Why it matters

- **A verifiable, non-Goodharted quality signal for math.** Correctness is already verifiable via the proof checker; *interestingness* is the missing rubric between "correct" and "worth stating". Length ratio is a cheap proxy that empirically tracks downstream reuse.
- **Enables self-expanding libraries.** Given a proof-search stack and this metric, an automated theorem prover can prioritise its own conjectures without human-supplied targets — a concrete step toward AlphaProof-style closed loops.
- **Transferable idea.** The proof-difficulty-conditioned-on-premises primitive is reusable — for auto-formalization, curriculum design, or as a reward-shaping signal in reasoning RL.

## Gotchas & tricks

- **Statement length is game-able.** A generator can inflate proof length by producing redundant lemmas or shorten statements by hoisting hypotheses into definitions. Constrain the score with a size floor on the statement.
- **Correlation with downstream utility is measured on Mathlib.** Interestingness in domain-specific libraries (physics, formal methods) may need different length proxies.
- **Difficulty predictor is a reward model** — subject to reward hacking. Ensembles or independent verification with an auxiliary predictor help.
- **Not a benchmark by itself.** It's a *metric* for ranking; standard benchmarks (MATH-500, AIME) still measure competence on fixed problems.

## Sources

- Paper: *Learning to Discover Interesting Mathematics* — Patel, Rammal, Hayat, Munos, Kempe (Meta / NYU), 2026, [arXiv 2609.28603](https://arxiv.org/abs/2609.28603) — introduces the proof/statement ratio, the 27B difficulty predictor, and the self-expanding-library recipe.
