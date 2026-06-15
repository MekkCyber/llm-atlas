# Generative Verifier

*Depth — a learned verifier that generates a verification/critique, not just a scalar score, engineered for low false-positive rate.*

**TL;DR:** A generative verifier is an LM trained to *produce* a verification trace (typically a chain-of-thought audit of a proposed solution) and then output a verdict. Unlike an [ORM](orm.md) (scalar score) or [PRM](prm.md) (per-step score), it leverages full generative capacity to find errors. Frontier reasoning systems engineer the verifier for **low false-positive rate** ("defense in depth") so it can be the linchpin of population-level test-time scaling without being gamed by reward hacking. Demonstrated at frontier scale by MaxProof on top of MiniMax-M3 — IMO 2025 gold and USAMO 2026 gold from the generator+verifier loop.

**Prereqs:** [orm.md](orm.md), [../rlvr.md](../rlvr.md), [../grpo.md](../grpo.md)
**Related:** [prm.md](prm.md), [population-test-time-scaling.md](population-test-time-scaling.md), [long-cot-rl.md](long-cot-rl.md), [../_rewards.md](../_rewards.md)

---

## What it is

The verifier as a *generative* model in its own right. Where ORMs collapse the verdict to a scalar head, a generative verifier writes out a verification trace — checking each step, flagging gaps, attempting counter-examples — and only then emits its verdict (correct / incorrect, sometimes with a confidence band).

The structural commitment is **defense in depth**: the verifier is engineered so that its **false-positive rate** (incorrectly claiming a wrong proof is right) is far smaller than its false-negative rate. Population-level [test-time scaling](population-test-time-scaling.md) relies on this asymmetry — the search loop generates and re-generates until *something* survives verification, so a single incorrectly-accepted proof would poison the entire output.

## How it works

### Capabilities trained into one model

Three closely-related capabilities are typically trained jointly (often via SFT followed by RL):

1. **Proof generation** — the standard reasoning policy.
2. **Proof verification** — given a problem and a candidate proof, produce a verification trace and verdict.
3. **Critique-conditioned repair** — given a proof and a verification critique, produce a corrected proof.

All three roles use the same model weights, distinguished only by prompt format. This is what makes the test-time loop coherent.

### Engineering low false-positive rate

- **Calibrated training data.** Negative examples (subtly wrong proofs) are mined adversarially — both from policy rollouts that pass shallow checks but fail formal ones, and from synthetic perturbations of correct proofs.
- **Multi-stage verification.** Defense-in-depth means the verifier runs multiple complementary passes (step-level, lemma-level, holistic) and only accepts when all pass.
- **Calibration against external ground truth.** During training, verifier accuracy is measured against rule-based or formal-proof checkers where available, so the verifier's verdicts are anchored.

### At test time

The verifier is the gate in a generator → verifier → refiner → ranker loop ([population-test-time-scaling.md](population-test-time-scaling.md)). Acceptance / rejection by the generative verifier drives both the loop's termination and the ranker's tournament selection.

## Why it matters

- **Headline frontier-math results.** MaxProof + MiniMax-M3 reaches **IMO 2025 35/42** and **USAMO 2026 36/42** — gold-medal threshold on both — and the *verifier* is the load-bearing component.
- **Pattern transfers beyond math.** Any domain with a generative verification semantics (code, formal proofs, structured outputs) can use the same recipe.
- **Defense-in-depth is the right framing for verifiers in RL.** A learned verifier with a high false-positive rate gets gamed by [GRPO](../grpo.md); engineering for low FPR shifts where the brittleness lives.

## Gotchas & tricks

- **False-positive rate dominates failure modes.** A verifier that's permissive 1% of the time will accept reward-hacked proofs ~1% of the time. Population search amplifies this — keep FPR tiny.
- **Critique-conditioned repair is the connective tissue.** Without it, the verifier's "this is wrong" signal is wasted — the generator never learns how to act on the critique. Train repair jointly.
- **Calibrate on adversarial negatives.** Easy negatives (random wrong answers) under-train the verifier; subtly wrong proofs (off-by-one, missing case) are what get accepted at test time.
- **Don't use as a dense RL reward without rule-based grounding.** Combine with rule-based outcome rewards ([rlvr](../rlvr.md)) so the verifier doesn't drift; reserve the generative verifier for offline filtering and test-time scaling.
- **Verifier-as-judge ≠ generative verifier.** A judge LM that scores a single response with a number is closer to an ORM; a generative verifier writes its work and uses the trace to decide.

## Sources

- Paper: *MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling* — Zhang et al., MiniMax, 2026 — [arXiv:2606.13473](https://arxiv.org/abs/2606.13473).
- Background: [orm.md](orm.md), [prm.md](prm.md), [../_rewards.md](../_rewards.md).
