# Generative Reward Model
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reward / verifier model that scores candidates by **generating tokens** in the LLM's own vocabulary rather than emitting a scalar. The score comes from log-probs (or a small parametric readout) over a fixed rubric-like continuation. Yields interpretable, localizable feedback that beats boolean-verdict solvers and generalizes across formal-translation styles.

**Prereqs:** [_rewards](_rewards.md)
**Related:** [reasoning/orm](reasoning/orm.md), [reasoning/prm](reasoning/prm.md), [cot-reward-model](cot-reward-model.md)

---

## What it is

A traditional reward model or verifier is a discriminative head (or an external solver) producing a scalar or boolean per candidate. A **generative reward model** (GenRM) instead prompts an LLM with the candidate and a rubric, then reads the score off the *generated tokens* — either directly ("Score: 7") or via log-prob margins between labeled continuations ("Yes" vs "No" in a critique). Because the score is computed in the same token space as the model that produced the candidate, error locations show up as high-loss regions in the generation, giving directable feedback.

GenV, applied to autoformalization verification, converts a Z3 equivalence oracle into a generative scorer that also points at the erroneous sub-expression.

## How it works

- **Prompt template.** `<candidate> ⇒ <rubric / question / oracle-informed cue>`. The rubric is the "reward channel".
- **Score readout.** Two common choices:
  - **Log-prob margin.** Score = $\log p(\text{"yes"}\,|\,\ldots) - \log p(\text{"no"}\,|\,\ldots)$, or an average over rubric fields.
  - **Chain-of-thought scoring.** Let the model generate a critique, then read a final score token. Higher variance but more interpretable.
- **Training.** Supervised on (candidate, verdict) pairs — the oracle's yes/no becomes the target for the generation. In GenV, the oracle is a Z3 equivalence check on a paired formal statement.
- **Localization for free.** Because the LLM generates over the candidate, per-token log-probs (or attention over the candidate) point to the sub-expression the model finds unfaithful — no separate error-localization module.

## Why it matters

Autoformalization pipelines were quietly failing via **Verdict-Preserving Unfaithfulness**: solver-passable formal translations that were not reference-equivalent. GenV catches this because its supervision is reference-equivalence, not verdict. More broadly, generative reward models replace opaque black-box checkers with something differentiable, transferable, and diagnosable — a strictly better fit for RL post-training loops that need dense, informative feedback.

Reported: 0.961 AUROC on verification; transfers across translators and formal styles; downstream task accuracy improves by ~11 percentage points.

## Gotchas & tricks

- **Rubric matters more than model size.** A well-designed rubric (rich yes/no questions with the right vocabulary) outperforms a bigger LLM on the same task.
- **Log-prob margin is cheaper than CoT scoring** and less noisy at small scales; CoT scoring becomes worth it when candidates are long or the rubric has many criteria.
- **Beware self-preference.** A GenRM built from model M scoring outputs from model M often prefers its own style. Mix training data across models.
- **Localization ≠ correctness.** The model may point at the wrong sub-expression; treat localization as a hint, not a verdict.
- **Combine with a solver, don't discard it.** GenV catches what the solver misses (unfaithfulness); the solver still catches what the GenRM misses (structural equivalence). The paper's best pipeline uses both.

## Sources

- Paper: *Beyond Solver Verdicts: Generative Reward Models for Autoformalization* — Singh, Ganguly, Goel, et al., 2026 — https://arxiv.org/abs/2609.11085
