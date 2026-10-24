# Provenance of "Wait" in Reasoning Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Tracing where the "wait…" token actually comes from in R1-style reasoning models. Across open model training histories, **zero-RL from a base that doesn't contain "wait" almost never produces it**, even while accuracy still rises substantially. "Wait" enters the model through **data** — mid-training on reasoning traces or SFT on human-written chains — and RL merely *reshapes* an existing marker. Also: LLM judges can reverse the human-labelled direction of effect on reasoning-marker metrics (HKUST · Huawei Noah's Ark Lab · PKU, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../post-training/rlvr.md](../post-training/rlvr.md), [../pre-training/mid-training.md](../pre-training/mid-training.md)

---

## What it is

An interpretability / methodology paper on reasoning-marker tokens. "Wait", "Let me reconsider", and other reflective tokens are frequently cited as evidence that RL induced reasoning. The paper asks the empirical question: did RL actually introduce them, or were they already there?

## How it works

- **Setup.** Collect open models with disclosed training histories (base → mid-training → SFT → RL). Measure "wait" frequency at each stage.
- **Zero-RL arms.** For bases *without* "wait", run RLVR-style zero-RL (no SFT, verifiable rewards only). Measure both accuracy and "wait" frequency.
- **Mid-training / SFT arms.** For bases with mid-training or SFT on reasoning traces, measure "wait" introduction at each stage.
- **Judge comparisons.** Rate reasoning quality with (a) human annotators on a controlled subset and (b) LLM judges on the same subset; compare the direction of effect of training interventions.

## Why it matters

- **"Wait" is a provenance signal, not an RL signal.** Zero-RL from "wait"-free bases almost never introduces the token, even while accuracy climbs. The frequency of "wait" is not evidence of RL-induced reasoning; it's evidence of *data* provenance.
- **Methodological warning for the field.** Papers claiming RL-induced reflective behaviour via marker frequencies need to control for mid-training / SFT provenance.
- **Judge reversal is damning.** LLM judges can give the opposite direction from humans on reasoning-marker metrics. Any pipeline that uses an LLM judge for reasoning quality should calibrate against humans — the direction, not just the magnitude, may flip.
- **RL reshapes rather than creates.** For bases that already contain "wait", different RL recipes change *how often* and *where* it appears, but it isn't invented from scratch.

## Gotchas & tricks

- **Scope: marker-level claim, not reasoning claim.** The paper is specifically about surface tokens. Deeper reasoning behaviours (planning, verification) may still be RL-driven; the marker frequency just can't be used as proof.
- **"Wait" is not universal.** Different languages and tokenisers emit different markers; the Chinese reasoning-model literature has its own set.
- **Zero-RL comparison requires clean bases.** Finding a base that genuinely has not seen "wait" at any pretraining stage is non-trivial; some apparent zero-RL successes may have had trace exposure.
- **Does not refute RL for reasoning.** The paper's claim is narrow: markers don't prove RL did anything. Accuracy still rises under RL in the paper's own measurements.

## Sources

- Paper: *Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models* — Kou et al., HKUST · Huawei Noah's Ark Lab · PKU, 2026 — [arXiv:2610.30882](https://arxiv.org/abs/2610.30882).
