# LeanPRM
*Depth — process reward model trained from Lean-verified formal proofs.*

**TL;DR:** PRM800K-style process reward models rely on expensive human step-labels. LeanPRM replaces humans with Lean: a model writes an informal natural-language proof and a matching Lean proof; Lean checks the formal steps; a second model strips failures caused by formal-syntax issues; what remains is per-step correctness labels for the natural-language proof. The resulting 10,634-proof / ~152K-step corpus trains a 4B PRM that beats a PRM800K-trained 4B PRM on step accuracy and best-of-N selection.

**Prereqs:** [prm.md](prm.md), [orm.md](orm.md), [cot-reward-model.md](../cot-reward-model.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [mcts.md](mcts.md)

---

## What it is

A formal-verification-as-labeler recipe for process supervision. Instead of asking annotators to grade each line of a math proof, LeanPRM asks a stronger model to write a Lean-matching version of each line, lets Lean check it, and lets a filter model remove non-mathematical failures.

## How it works

1. For each informal problem, prompt a strong model to produce an informal step-by-step proof plus a matching Lean proof, step-aligned.
2. Run Lean's kernel on the formal proof. Each formal step passes or fails.
3. Pass failures to a filter model that labels each failure as "formal-syntax issue" (ignore) or "mathematical error" (keep).
4. Attach the surviving per-step pass/fail labels to the natural-language steps.
5. Train a transformer PRM to predict step-correctness on the labeled corpus. Score candidate NL proofs step by step at inference.

## Why it matters

PRM800K is still the de-facto human-labeled process-reward corpus. If Lean coverage is enough to scale, the ceiling moves from annotator budget to formal-theorem corpus size — potentially orders of magnitude more. Lean also catches errors humans miss and is cheap to re-run as Lean's library grows.

## Gotchas & tricks

- Depends on a model that can write matching Lean — currently a small, strong fine-tune. Weaker generators produce unverifiable or vacuous Lean.
- "Formal-syntax issue" classifier is itself a model; its false negatives silently poison labels. Spot-check with humans on a validation slice.
- Covers only mathematical domains Lean's library supports well (olympiad-level algebra/number theory/combinatorics strong; analysis thinner).

## Sources

- Paper: *LeanPRM: Lean-Grounded Process Supervision for Natural Language Proof Verification* — Rahman, Shen, Shiffer, Gabriel, Izmailov — UCLA/NYU, 2026 — [arXiv:2610.17697](https://arxiv.org/abs/2610.17697)
