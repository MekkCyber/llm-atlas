# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training paradigm where the student generates its **own** rollouts and is trained against a teacher's per-token feedback on *those* rollouts — as opposed to offline distillation on teacher-generated text. Getting the KL direction right matters: **forward KL** (teacher-weighted) is stable under asynchronous rollouts; **reverse KL** (student-weighted) is more sample-efficient but fragile. Increasingly the default finetune-from-frontier recipe for reasoning models where quality of the rollout distribution matters as much as its next-token target.

**Prereqs:** [_post-training.md](./_post-training.md), [fine-tuning/README.md](./fine-tuning/README.md)
**Related:** [rlvr.md](./rlvr.md), [asyncopd.md](./asyncopd.md), [grpo.md](./grpo.md)

---

## What it is

Classical distillation samples data from the teacher; the student learns to imitate. On-policy distillation instead samples from the *student*, and the teacher acts as a per-token feedback function on the student's outputs. This is analogous to RL — the student's rollouts drive the update — but the reward is teacher log-probs, not a scalar. It sits between SFT (offline data) and RL (scalar reward, on-policy) as a middle-ground post-training regime.

## How it works

**Basic loop.**
1. Student generates a rollout `y` given prompt `x`, one token at a time.
2. Teacher computes per-token log-probs on the same sequence.
3. Loss is a KL between student and teacher distributions over the vocabulary at each position.
4. Update the student. Repeat.

**KL direction is the key design choice.**

- **Forward KL:** `KL(teacher || student)` — teacher-weighted. Penalises places where the teacher assigns high probability but the student does not. Covers the teacher's whole distribution ("mean-seeking"). Robust to stale rollouts.
- **Reverse KL:** `KL(student || teacher)` — student-weighted. Penalises places where the student assigns high probability but the teacher does not. Mode-seeking; more sample-efficient but sensitive to rollout staleness (see [asyncopd.md](./asyncopd.md)).

**Teacher-score caches.** Storing the teacher's *full-vocabulary* logits per token during the student rollout is expensive at frontier scale; in practice one stores a *sampled* or *sparse* subset, which trades bias for variance and creates a bias-variance dial for reverse-KL estimators.

**Where it fits in a post-training pipeline.**
- After SFT, before RL: use OPD to align rollouts with teacher style/quality on real distributions.
- Alongside RL: some pipelines interleave OPD steps with RL steps to stabilise the reward signal.
- Standalone: for domains where a scalar reward is unavailable but a stronger teacher is (e.g. distilling a reasoning frontier model into a smaller one).

## Why it matters

- **Bridges the SFT-RL gap.** SFT gives token-level signal but no coverage of the student's own errors; RL gives coverage but only a scalar reward. OPD gives token-level signal *on the student's own distribution*.
- **First-class option for frontier-to-smaller distillation.** Recent reasoning post-training pipelines (see multi-teacher OPD variants) rely on OPD to move R1/o1-class capability into smaller student sizes.
- **Systems shape matches RL.** Rollout workers, learner processes, staleness — same infrastructure. Which lets labs reuse their RL stack.

## Gotchas & tricks

- **Reverse KL under staleness collapses.** If your rollouts are hours old and you're using reverse KL, expect training instability. Either switch to forward KL or use the OPD-specific "recompute reverse KL at learner time" surrogate (see [asyncopd.md](./asyncopd.md)).
- **Teacher-cache granularity is a real knob.** One sample per position is high variance; storing all logits is prohibitive; multi-sample MC estimators are the modern compromise.
- **Domain mismatch between teacher and student hurts.** OPD assumes the teacher can score *the student's* distributions meaningfully. A teacher that's too far above the student may return uninformative gradients (nearly-flat distributions on the student's outputs).
- **Not a replacement for RLVR.** OPD lacks a hard verification signal — for math/code where correctness is checkable, RL with rule-based rewards still dominates for the last mile.

## Sources

- Paper: *AsyncOPD: How Stale Can On-Policy Distillation Be?* — Kang et al., 2026 — [arXiv:2606.24143](https://arxiv.org/abs/2606.24143). First systematic study of OPD staleness; the KL-direction and cache-granularity analysis this file summarises.
- Related: OPD is used throughout multi-teacher distillation pipelines described in agent tech reports (e.g. Agents-A1).
