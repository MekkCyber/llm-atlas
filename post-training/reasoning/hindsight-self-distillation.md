# H²SD (Hybrid Hindsight Self-Distillation)
*Depth — a trajectory-conditional self-distillation loss for RLVR reasoning.*

**TL;DR:** RLVR gives one scalar reward per trajectory, which under-uses per-token information. H²SD densifies supervision by using the *same model* as a teacher, differently by trajectory correctness: on **successes**, the teacher receives the confirmed-correct student response with a rephrasing prompt, and its per-token probabilities modulate update *magnitudes* without changing direction. On **failures**, the teacher is conditioned on a reference hint + verified answer, and the student minimizes reverse KL to that teacher — an explicit correction *direction*.

**Prereqs:** [../rlvr.md](../rlvr.md), [../grpo.md](../grpo.md)
**Related:** [../rejection-sampling.md](../rejection-sampling.md), [long-cot-rl.md](./long-cot-rl.md)

---

## What it is

Reasoning-oriented RLVR needs dense per-token signal but resists direct distillation:
- **On-policy distillation (OPD)** demands a separate stronger teacher and a shared vocabulary.
- **On-policy self-distillation (OPSD)** removes the teacher dependency by conditioning the same model on privileged info — but naïve reverse-KL matching leaks information and destabilizes training.
- **RLSD** avoids direct matching, using the teacher signal only to modulate update magnitudes — but never gives the student an explicit correction direction when it's wrong.

H²SD is a *hybrid*: the teacher plays different roles depending on whether the trajectory succeeded or failed.

## How it works

For a rolled-out trajectory with binary verifier outcome `r ∈ {0, 1}`:

**Successful trajectory (r = 1).**
1. Feed the confirmed-correct student response to the teacher with a **rephrasing instruction** (e.g. "restate the reasoning").
2. Read out the teacher's per-token probabilities on the *original* response tokens.
3. Use them to **modulate update magnitudes** — do not change the sign or direction the RL reward already implies.

**Failed trajectory (r = 0).**
1. Condition the teacher on a **reference hint** containing key reasoning steps plus the verified final answer.
2. Minimize **reverse KL** from the student's distribution to this teacher's distribution.
3. This provides an explicit correction *direction* — where the student's reasoning should have gone.

The two branches share teacher weights (self-distillation) but differ in conditioning and in how the teacher's signal enters the loss.

## Why it matters

- **Cleanly separates two failure modes** of self-distillation: magnitude noise on correct rollouts and missing direction on wrong ones.
- **Consistent gains** across reasoning benchmarks over RLVR, OPSD, and RLSD baselines, with stable optimization and no throughput penalty relative to plain self-distillation.
- **Framing durability.** Even if the exact algorithm gets superseded, "treat successes and failures with different teacher roles" is a template that will stick.

## Gotchas & tricks

- Reverse KL from student to a hint-conditioned teacher can be aggressive — the paper is careful about scaling; naïve implementations diverge.
- Rephrasing instruction on the success branch is meant to elicit *teacher probabilities on the same tokens*, not to change them; getting the prompt wrong leaks the correct answer back into the loss.
- Hint construction on the failure branch is a labeling problem in disguise; the paper assumes a verifier can supply reasoning steps or that they can be synthesized.

## Sources

- Paper: *H²SD: Hybrid Hindsight Self-Distillation* — Ma et al., 2026 — [arXiv:2607.18955](https://arxiv.org/abs/2607.18955)
