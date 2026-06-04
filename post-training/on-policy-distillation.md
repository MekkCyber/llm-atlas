# On-Policy Distillation (OPD)

*Depth — distill a teacher into a student using the student's *own* rollouts as the supervision target distribution.*

**TL;DR:** Standard distillation (offline / off-policy) fits the student to teacher outputs drawn from the teacher's distribution. OPD instead samples sequences from the *student* and supervises each student-generated token with the teacher's distribution at that context. This removes the train-time / inference-time distribution shift that plagues offline distillation, giving denser-than-RL credit assignment while staying on the student's reachable trajectories.

**Prereqs:** [_post-training.md](./_post-training.md), [_rl.md](./_rl.md), [rejection-sampling.md](./rejection-sampling.md)
**Related:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md), [dpo.md](./dpo.md)

---

## What it is

OPD is the third option in the post-training space between SFT and RL:

| Method | Source of trajectories | Supervision |
| --- | --- | --- |
| SFT (off-policy) | Teacher / static dataset | Cross-entropy on teacher tokens |
| OPD | **Student** (on-policy rollouts) | Per-token teacher distribution at student contexts |
| RL | Student | Sparse reward at end of trajectory |

OPD fixes the off-policy bias of SFT (the student never sees its own mistakes) and the credit-assignment problem of RL (one scalar reward must explain a long trajectory) by using the teacher as a **dense, on-policy critic**.

## How it works

```
for each prompt q:
    sample o ~ student(· | q)            # on-policy rollout
    for each token o_t in o:
        target = teacher(· | q, o_<t)    # teacher's full distribution
        loss   += KL(student(· | q, o_<t)  ||  target)
                  # reverse-KL: student matches teacher on student-visited states
```

Three failure modes everyone hits:
- **Distribution mismatch.** If the student is much weaker than the teacher, student rollouts land in regions where the teacher's distribution is near-uniform or weird — the reverse-KL gradient becomes noisy and can diverge. *Trust Region OPD* (TrOPD, Samsung 2026) only updates inside a trust region of reliable teacher supervision and uses forward-KL or masking on outlier tokens.
- **Logit access requirement.** Vanilla OPD needs the teacher's per-token logits — excludes closed APIs. *OmniOPD* (Meta 2026) replaces logit matching with chunk-level semantic similarity scored by Monte Carlo teacher rollouts, plus a peak-entropy scheduler that only audits the student at high-uncertainty forks.
- **Degenerate-pattern amplification.** Dense token matching can lock in repetition loops if the teacher's top-1 token sequence cycles. KL anchors to the base model (used by both OmniOPD and most production OPD pipelines) prevent collapse.

## Why it matters

- **Bridges SFT and RL.** Denser signal than RL, less off-policy than SFT — the empirically dominant move when a stronger teacher exists.
- **Compression recipe.** A student matching teacher distributions on its own trajectories beats SFT-on-teacher-outputs at fixed compute, because the student trains on inputs it will actually see at inference.
- **Composes with RL.** Used as a warm-start before RL (TrOPD), as the *consolidation* phase in continual-learning schemes (Sleep, Google 2026: "Knowledge Seeding" distills a smaller past-self upward into a larger current network via OPD+RL imitation), and as a step in agentic post-training.

## Gotchas & tricks

- **Forward vs reverse KL.** Reverse-KL on student rollouts is mode-seeking — fine when the teacher is confident, brittle when not. Forward-KL on teacher rollouts is mode-covering — use it as the *off-policy guidance* branch (TrOPD) to nudge the student toward reliable teacher regions.
- **Selective supervision saves teacher cost.** Auditing only at high-entropy student tokens (OmniOPD's peak-entropy scheduler) cuts teacher calls by an order of magnitude with no measurable quality loss.
- **Logit-free is API-friendly.** Chunk-level semantic similarity against Monte Carlo teacher rollouts lets you use Claude / Gemini / GPT as teachers via API.

## Sources

- *Trust Region On-Policy Distillation* — Xing et al., Samsung Research, 2026 — [arXiv:2606.01249](https://arxiv.org/abs/2606.01249).
- *OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification* — Zhou et al., Meta, 2026 — [arXiv:2606.01476](https://arxiv.org/abs/2606.01476).
- *Language Models Need Sleep* — Behrouz & Mirrokni, Google, 2026 — [arXiv:2606.03979](https://arxiv.org/abs/2606.03979) — Knowledge Seeding as upward-OPD-plus-RL.
- Earlier on-policy distillation lineage in image classification (Hinton-style) and in *Generalized Knowledge Distillation* (Agarwal et al., 2024) for language models.
