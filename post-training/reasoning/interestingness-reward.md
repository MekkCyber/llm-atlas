# Interestingness Reward for Open-Ended Math
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An RL signal for **open-ended mathematical discovery**: define a theorem's "interestingness" as the ratio between the length of its formal proof and the length of its statement, given a fixed background library, and reward the model for finding conjectures whose interestingness sits in a productive band. Combines a formal-verification outcome reward with an interestingness score computed by a 27B difficulty-predicting model. Reduces overlap with existing libraries from **91.9% → 30.6%**, i.e. the system actually finds novel theorems.

**Prereqs:** [rlvr](../rlvr.md), [rl-prompt-curation](../rl-prompt-curation.md)
**Related:** [long-cot-rl](long-cot-rl.md), [../grpo](../grpo.md), [../_rewards](../_rewards.md)

---

## What it is

Naive RL for math theorem proving optimizes "prove more theorems from a training set" and collapses onto rediscovery — the model just reproves already-known facts. The paper argues this is because there is no signal for *which conjectures are worth proving*. Interestingness as a scalar gives the RL loop a target that rewards novelty and depth simultaneously.

## How it works

**Definition.** For a candidate theorem $T$ with statement $s(T)$ and shortest known formal proof $p(T)$ against a background library $\mathcal{L}$:

$$
\mathrm{Interestingness}(T \mid \mathcal{L}) \;=\; \frac{\mathrm{len}(p(T))}{\mathrm{len}(s(T))}
$$

- Statements are much shorter than proofs → high ratio ⇔ the theorem is short-to-state but hard-to-prove given $\mathcal{L}$ (the classical "elegant, deep" pattern).
- Ratio depends on $\mathcal{L}$: as the library grows, previously interesting theorems become less interesting (their proofs shorten), pushing exploration toward new fronts.

**Difficulty predictor.** A 27B model is trained to predict $\mathrm{len}(p(T))$ from $s(T)$ and $\mathcal{L}$ without actually finding the proof — this makes interestingness computable at candidate-generation time and lets the RL loop select conjectures before spending compute on proof search.

**RL loop.**
1. Propose candidate conjectures (from a generator LLM).
2. Predict interestingness with the 27B model; keep the top band.
3. Attempt a formal proof; verify with the proof assistant.
4. Reward = verified × interestingness-in-band.
5. Add proven theorems to $\mathcal{L}$; recompute interestingness for the next round.

## Why it matters

- **Concrete reward signal for open-ended math discovery** — the missing piece for auto-expanding a formal library beyond human-set targets.
- **91.9% → 30.6% library-overlap** on evaluation, i.e. the system genuinely finds novel theorems, not just proves known ones.
- **Curriculum by construction.** The interestingness band acts as a difficulty filter — analogous to [rl-prompt-curation](../rl-prompt-curation.md) for reasoning RL — but derived from the library's current state instead of static difficulty tags.

## Gotchas & tricks

- The ratio is well-defined only relative to a fixed library — comparing scores across projects requires normalizing to the same $\mathcal{L}$.
- Length metrics on formal proofs can be gamed by tactic bloat; use a normalized "proof term" length rather than raw script length.
- The difficulty predictor is a single point of failure; miscalibration means the RL loop wastes compute proving trivial "interesting-looking" claims. Retrain periodically as $\mathcal{L}$ expands.
- Works best paired with strong outcome verification (Lean/Isabelle) — the interestingness reward on top of an unreliable prover collapses to noise.

## Sources

- Paper: *Learning to Discover Interesting Mathematics* — Patel, Rammal, Hayat, Munos, Kempe — NYU / Meta AI, 2026 — [arXiv:2609.28603](https://arxiv.org/abs/2609.28603).
