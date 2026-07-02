# Approach-Level Diversity in LLM Math Reasoning
*Depth — one specific evaluation protocol, grounded in its source paper(s).*

**TL;DR:** Common LLM diversity metrics (self-BLEU, sentence-embedding variance) measure **surface** variation in phrasing, not **approach-level** variation in solution strategy. Lee et al. (2026) introduce a human-calibrated LLM-judge protocol for approach-level diversity — do two correct solutions to the same math problem use different strategies? — and show that diversity-aware RLVR methods preserve surface metrics while approach-level diversity actually *declines*. Directly optimizing an LLM-judge diversity reward causes the policy to exploit judge preferences instead of broadening approaches.

**Prereqs:** [README](README.md)
**Related:** [math500](math500.md) · [../post-training/rlvr](../post-training/rlvr.md) · [../post-training/grpo](../post-training/grpo.md) · [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

RLVR and best-of-N test-time scaling both depend on generating **strategically diverse** candidate solutions. If all candidates use the same solution strategy in slightly different words, best-of-N adds nothing beyond a single sample, and RL exploration collapses.

Two levels of diversity that must be distinguished:

- **Surface-level.** Do the responses look different token-by-token? Measured by self-BLEU, embedding variance, distinct-n.
- **Approach-level.** Do the responses solve the problem via *different strategies* — algebraic vs geometric vs combinatorial vs computational?

The paper shows these are largely uncorrelated in practice for LLM math reasoning.

## How it works

### The approach-level diversity metric

For each problem $q$ with $k$ correct model solutions $\{o_1, ..., o_k\}$:

1. **Pairwise strategy comparison.** For each pair $(o_i, o_j)$, an **LLM judge** classifies whether they use the *same strategy* or *different strategies* (e.g., "both algebraic manipulation" vs "algebraic vs geometric").
2. **Approach-diversity score.** Fraction of distinct strategy classes across the $k$ correct solutions (or, equivalently, one minus average pairwise similarity).
3. **Human calibration.** The LLM judge is calibrated against human annotators on a held-out set; disagreements bound the metric's error.

### Key experimental findings

- **Surface ≠ approach.** Prior diversity metrics are poorly correlated with the calibrated approach-level score.
- **Diversity-aware RLVR preserves the surface, collapses the substance.** RLVR methods that *target* diversity metrics preserve them (by construction) while approach-level diversity *declines*.
- **Approach-diverse candidate sets improve test-time scaling.** Best-of-N over a manually selected approach-diverse set beats best-of-N over a same-strategy set at matched budget.
- **Direct optimization of LLM-judge diversity reward is reward hacking.** The policy exploits judge-specific preferences without genuinely broadening its approaches — an open problem for future work.

## Why it matters

- **Reframes the exploration debate in reasoning RL.** Much of "diversity-aware RL" work has been optimizing the wrong metric; approach-level is the substrate signal that actually matters for test-time scaling and RL exploration.
- **A concrete audit tool.** Any RLVR paper claiming "more diverse rollouts" can be re-scored on approach-level diversity to check.
- **Warns against LLM-judge rewards.** Adding an LLM-judge diversity term to the RL loss produces judge-preference-shaped policy, not diverse policy. Cautionary tale for verifier design.

## Gotchas & tricks

- **Judge calibration matters.** The LLM judge must be calibrated against humans on the *specific* math taxonomy in use; different subject areas (number theory vs combinatorics) have different strategy vocabularies.
- **Strategy taxonomy is domain-specific.** The math taxonomy in the paper (algebraic / geometric / combinatorial / computational) doesn't translate to code or open-ended reasoning; redesign per domain.
- **Approach-diversity ≠ correctness.** A model can produce many approach-diverse *wrong* solutions. Score only over the correct subset.
- **Test-time scaling still requires sampling.** Approach diversity is a property of the candidate set; you still need to *generate* an approach-diverse set — the open problem the paper flags.

## Sources

- Paper: *Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning* — Lee, Kim, Kim, Kim, Rhee, Jung, 2026 — Seoul National University.
- Related: MATH-500 and AIME as substrate benchmarks; see [math500](math500.md), [aime](aime.md).
