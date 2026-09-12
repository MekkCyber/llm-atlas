# Generative Reward Model
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A class of learned scorers that supply **continuous, reference-free** rewards distilled from a symbolic oracle. Instead of returning a scalar preference, a generative reward model produces a scored *generation* whose sample-level statistics form the reward — the same head that generates critique text also emits the numeric score. Introduced for autoformalization ("Beyond Solver Verdicts"), where the oracle is an offline Z3-equivalence checker: the reward model reaches **0.961 AUROC** at judging whether a candidate formalization is reference-equivalent to a gold one, and generalizes across translators and formal styles.

**Prereqs:** [../_rewards](_rewards.md), [cot-reward-model](cot-reward-model.md)
**Related:** [reasoning/orm](reasoning/orm.md), [reasoning/prm](reasoning/prm.md), [rlvr](rlvr.md)

---

## What it is

Solvers (SAT/SMT/proof assistants) give a binary *verdict-preserving* signal — "candidate formalization $\phi$ produces the same solver verdict as gold formalization $\phi^*$." Verdict matches can happen coincidentally when $\phi$ is meaning-*wrong* but structurally similar ("Verdict-Preserving Unfaithfulness"). A generative reward model targets the harder **reference-equivalence** question — do $\phi$ and $\phi^*$ have identical models? — and produces a continuous score without needing $\phi^*$ at inference.

## How it works

**Offline oracle.** For a corpus of (natural-language problem, gold formalization, candidate formalization) triples, run an SMT equivalence check via Z3 (or a proof-assistant analogue): $\text{Oracle}(\phi, \phi^*) \in \{\text{equiv}, \text{not-equiv}\}$.

**Distillation.** Train an LLM $R_\theta$ to predict the oracle label from `(problem_text, candidate_formalization)` alone — **the gold formalization $\phi^*$ is not shown at inference**. Cross-entropy loss on the equivalence label; optionally jointly train a critique head that generates a rationale.

**Continuous score.** At inference, $R_\theta$ produces a probability of reference-equivalence (or an aggregated log-prob over the generative rationale). That real-valued score becomes the RL reward for a downstream translator policy.

**Reward integration.** Plug into any RLVR/GRPO/PPO loop: sample formalization candidates from the translator, score with $R_\theta$, run policy update. Because the score is continuous and reference-free, the loop needs neither gold formalizations at rollout time nor a running solver.

## Why it matters

- **Semantic reward for autoformalization.** Verdict matching rewarded lucky solver flukes; reference-equivalence rewards *meaning*.
- **Reference-free.** No gold formalization at inference — a scalable substitute for solver-in-the-loop RL.
- **Generalizes across formal styles.** The paper reports transfer across translators and target logics (SMT-LIB, Lean, etc.), because the reward looks at natural-language + candidate rather than tying to a fixed formalism.
- **Template for other neurosymbolic loops.** Any offline symbolic oracle can be distilled this way: theorem provers, SQL semantic-equivalence checkers, unit-test executors on program pairs.

## Gotchas & tricks

- **Oracle coverage.** Z3-equivalence times out on hard formulas; label those as "unknown" rather than "not-equiv" to avoid injecting false negatives.
- **Distribution shift on the translator.** As the RL translator improves, its outputs drift from the distillation distribution; refresh the reward model periodically on fresh translator samples.
- **Reward hacking through rationale.** If the critique head is used to compute the score (log-prob of "equivalent"), the policy can game the rationale generator; freeze it or add a rationale-consistency check.
- **Not a substitute for the oracle at eval time.** Use $R_\theta$ for training reward; still evaluate final model outputs with the true oracle.
- **AUROC ≠ RL usefulness.** A high-AUROC scorer with a saturated distribution gives noisy gradients — check score calibration, not just discrimination.

## Sources

- Paper: *Beyond Solver Verdicts: Generative Reward Models for Autoformalization* — Singh, Ganguly, Goel, Torkamani, Han, Lilien, Erata, Chaudhary (Case Western / Yale / others), 2026 — [arXiv:2609.11085](https://arxiv.org/abs/2609.11085)
- Related: process/outcome reward models — sibling scorer classes with different resolution.
- Related: RLVR — the class of RL loops this reward slots into when the verifier is generative rather than symbolic.
