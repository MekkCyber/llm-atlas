# LeanPRM
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Process reward models (PRMs) historically depend on expensive per-step human labels (PRM800K-style) or Monte-Carlo rollout labels (Math-Shepherd). LeanPRM replaces the labeler with the **Lean proof assistant**: each natural-language proof step is translated into a Lean goal, and Lean's kernel decides whether that step is formally sound. Produces a 10,634-proof corpus with 152K step labels — zero human annotation — and a Qwen3-4B PRM trained on it beats PRM800K-trained verifiers on 4 of 5 benchmarks.

**Prereqs:** [prm.md](./prm.md), [orm.md](./orm.md)
**Related:** [../rlvr.md](../rlvr.md), [../_rewards.md](../_rewards.md), [long-cot-rl.md](./long-cot-rl.md)

---

## What it is

A PRM data-generation pipeline grounded in a formal verifier. Instead of asking humans "is step $i$ correct?", LeanPRM asks the Lean theorem prover. The output is a corpus of (prompt, step-sequence, per-step correctness label) tuples, where each label is the output of a sound mechanical check — not a human judgment.

This is to PRM what RLVR is to outcome rewards: swap a learned/human labeler for a formal verifier that can't be gamed by surface patterns.

---

## How it works

1. **Translate** each NL step of a math proof into a Lean goal, using an LLM-based translator trained for the purpose.
2. **Check in Lean.** Run Lean's kernel on the translated step. If the step closes a subgoal (or produces a well-typed term consistent with the preceding state), label it correct; otherwise incorrect.
3. **Repeat per step** down the full proof. Mis-translations become "unverifiable" and are either dropped or treated as auxiliary signal.
4. **Train a PRM** on the resulting labeled corpus using the usual step-classification objective: Qwen3-4B as the base; cross-entropy per-step with aggregation (min, product, mean) at inference.

The authors also release **LeanPRM-Bench**, a 200-proof held-out benchmark with Lean-verified step labels, as a cleaner evaluation signal than PRM800K's human labels.

---

## Why it matters

- **Scaling human-free step labels.** PRM800K's 800K steps required ~a year of human annotation. Lean-grounded labels are limited by compute, not by budget.
- **Beats PRM800K-trained verifiers.** On 4 of 5 benchmarks the LeanPRM-trained Qwen3-4B PRM outperforms a same-size PRM trained on human labels — the formal signal is cleaner than human disagreement.
- **Better best-of-N selection.** On IMO-level problems the LeanPRM reranks candidate proofs more accurately than human-labeled PRMs, suggesting the formal signal transfers to problems far from the training distribution.
- **Parallels RLVR's trajectory.** RLVR replaced human preference RMs with rule verifiers for outcome rewards. LeanPRM is the step-level analogue — the same philosophical move, applied to PRMs.

---

## Gotchas & tricks

- **Translator quality is a floor.** Mis-translation of an NL step into Lean yields a wrong label. The paper handles this with conservative labeling (drop ambiguous steps), but the translator stays on the critical path.
- **Lean coverage is narrower than NL math.** Steps that require implicit reasoning or appeals to common knowledge may not translate at all. The corpus is skewed toward formally expressible proof styles.
- **Scales to other proof assistants.** Coq / Isabelle / Metamath should work with the same recipe. The paper focuses on Lean because of its mature library (`mathlib`).
- **PRM remains a reranker, not an RL reward.** The LeanPRM paper evaluates mostly in best-of-N; using it as a dense RL reward inherits the same reward-hacking risks DeepSeek-R1 found with Math-Shepherd-style PRMs (see [prm.md](./prm.md)).

---

## Sources

- Paper: *LeanPRM: Lean-Grounded Process Supervision for Natural Language Proof Verification* — Rahman, Shen, Shiffer, Gabriel, Izmailov, UCLA / NYU, 2026 — arXiv [2610.17697](https://arxiv.org/abs/2610.17697).
- Baseline: *Let's Verify Step by Step (PRM800K)* — Lightman et al., 2023.
- Baseline: *Math-Shepherd* — Wang et al., 2024.
