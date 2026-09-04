# On-Policy Distillation

*Depth — distilling a teacher LM using student-generated tokens and teacher log-probs at sampled positions only.*

**TL;DR:** Classical distillation trains the student to match the teacher's *full* next-token distribution on teacher-produced data — expensive, and off-policy. **Sampled-token on-policy distillation (OPD)** instead trains on the *student's* own generations, asking the teacher only for probabilities at the tokens the student sampled. Cheap, natural fit for reasoning tasks. But it comes with a diagnosed failure mode — **the diversity bottleneck**: pass@1 improves while pass@k plateaus. **IDA-OPD** (Yang et al., 2026) introduces a first-order local entropy influence proxy that separates entropy-expanding updates (keep) from entropy-contracting ones (shrink), fixing pass@k without touching the sampled-token infrastructure or requiring full-vocab teacher access.

**Prereqs:** [_post-training](_post-training.md), [rejection-sampling](rejection-sampling.md)
**Related:** [rlvr](rlvr.md), [reasoning/first-mistake-reward](reasoning/first-mistake-reward.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

A distillation family that trains the student on its own on-policy rollouts, requesting from the teacher only the log-probabilities of the tokens the student actually sampled. Contrasted with **off-policy full-vocab distillation** (teacher generates data, teacher gives full next-token distribution) and with **RL post-training** (no teacher, ground-truth reward).

Two objectives typically:

1. **Behavior cloning on sampled tokens** — increase student probability of teacher-preferred tokens.
2. **Reverse-KL / forward-KL on the sampled position only** — the sampled-token log-prob gap between teacher and student is the training signal.

## How it works

### The base OPD loop

For each training step:

1. Student samples a rollout from a prompt.
2. Teacher scores the sampled tokens (one forward pass; no full vocab probabilities needed).
3. Update the student to reduce the teacher–student log-prob gap at the sampled positions.

This is much cheaper than full-vocab forward-KL — no top-1000 teacher softmax per position — and it uses student trajectories, which matches the training distribution to what the student will see at inference.

### The diversity bottleneck

Empirically, OPD improves pass@1 but pass@k plateaus. The paper diagnoses this with **First-Order Local Entropy Influence** — a signed proxy that decomposes each update's effect on entropy into two factors:

- The **teacher–student log-prob gap** at the sampled token (how much the update pulls the distribution).
- The **student's local probability structure** (how peaked the current distribution is).

The sign of that product predicts whether the update *expands* entropy (good — preserves diversity) or *contracts* it (bad — makes the student collapse onto the teacher's mode). Empirically, entropy-contracting updates concentrate at "negative-influence positions" and drive the diversity collapse.

### IDA-OPD's fix

**Influence-Directed Adaptive OPD** (IDA-OPD):

- Keep entropy-expanding updates as-is.
- Replace entropy-contracting updates with **divergence-adaptive advantage shrinkage** — reduce the magnitude of updates that would collapse diversity, in proportion to the local KL divergence.

Only sampled-token teacher log-probs are needed; no full-vocab forward-KL. So the fix keeps the OPD cost profile but preserves pass@k.

### Empirical shape

On reasoning-oriented distillation:

- IDA-OPD consistently improves pass@k across benchmarks (inheriting teacher diversity).
- Matches the strongest full-vocab teacher-informed methods at **strictly lower cost**.
- Broadly maintains vanilla OPD's pass@1 (i.e. the fix doesn't sacrifice the primary metric OPD was already good at).

## Why it matters

- **Pass@k is the currency of downstream RL.** Once the student is used as an initialization for RL post-training, self-consistency, or best-of-N sampling, pass@k drives final accuracy. An OPD student that collapses to pass@1 wastes the diversity budget that later stages want to spend.
- **Cheap fix, no infra change.** Full-vocab forward-KL preserves diversity but costs a full teacher softmax per position — often infeasible for large teachers. IDA-OPD keeps the sampled-token profile.
- **Names an important failure mode.** Prior OPD papers reported pass@1 improvements without checking pass@k; this paper's contribution is partly diagnostic — future distillation ablations will need to report both axes.
- **Bridge to RL.** OPD sits between SFT and RL — an on-policy loop, but with a teacher rather than a reward. Understanding when it collapses diversity clarifies when to switch to RLVR / [first-mistake-reward](reasoning/first-mistake-reward.md).

## Gotchas & tricks

- **Report pass@k, not just pass@1.** The diversity bottleneck is invisible if only pass@1 is measured.
- **Small local KL ≠ safe update.** A tiny gap can still be entropy-contracting; use the signed First-Order Local Entropy Influence, not the KL magnitude alone.
- **The teacher's diversity is the ceiling.** OPD (with or without IDA) can only inherit diversity the teacher has. A near-greedy teacher gives a near-greedy student; if diversity matters, sample from the teacher at higher temperature during log-prob extraction.
- **Rollout batch composition matters.** Rollouts drawn only from correct prefixes underrepresent the negative-influence positions; sample enough imperfect rollouts to expose the diversity-collapsing updates.
- **Not a replacement for RL.** For tasks with a rule-verifiable reward, RLVR-style optimization can go beyond what any teacher's distribution allows. OPD is a strong initializer; RL is where the ceiling actually lifts.
- **Composes with [rejection-sampling](rejection-sampling.md).** OPD on rejection-sampled correct traces is a common practical stack; IDA's fix layers on top without changing that.

## Sources

- Paper: *Influence-Directed Distillation: Solving the Diversity Bottleneck in Sampled-Token On-Policy Distillation* — Run Yang, Runpeng Dai, Jie Sun, Jielei Zhang, Fan Zhou, Hongtu Zhu, Peiyi Li, Longwen Gao — 2026 — [arXiv:2608.29846](https://arxiv.org/abs/2608.29846).
- Related: *Distilling Reasoning Chains from LLMs* — foundational OPD-style distillation for reasoning.
- Related: [rlvr](rlvr.md) — the next stage in a typical distill → RL pipeline.
