# AsyncOPD
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **fully asynchronous** on-policy-distillation pipeline that decouples rollout generation from learner updates. Three technical claims: (1) **forward KL** tolerates stale rollouts; **reverse KL** doesn't. (2) For the reverse-KL case, **recomputing the KL under the current student at learner time** beats stabilisers borrowed from async-PPO. (3) Multi-sample Monte-Carlo estimators recover a workable bias-variance tradeoff when full teacher logits are too expensive to cache. Delivers 1.6×–3.8× throughput over synchronous OPD at matched accuracy.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)
**Related:** [_post-training.md](./_post-training.md), [rlvr.md](./rlvr.md)

---

## What it is

Synchronous OPD forces the learner to wait for every rollout to finish before applying an update — the same rollout-vs-learner bottleneck that motivated async RL. AsyncOPD is the first systematic treatment of what "async" costs for OPD specifically, and the mitigations that recover its accuracy under realistic staleness.

## How it works

**Async pipeline shape.**
- Rollout workers sample from the student policy `π_θ_gen`, snapshotted from the learner some number of updates ago.
- The learner uses these stale rollouts + the *current* teacher scoring to compute a KL loss and step.
- Staleness = number of learner updates between when the rollout was generated and when it's consumed.

**Three findings.**

**(1) KL direction changes the staleness problem.**
- **Forward KL:** `KL(teacher || student)` — teacher's mass over the vocabulary; the gradient direction is defined by *teacher* probabilities on stale rollouts, and this is robust to which student generated the tokens.
- **Reverse KL:** `KL(student || teacher)` — weighted by *student* probabilities; if the current student differs from the generating student, the estimator drifts.

**(2) Stabilise reverse KL by recomputing at learner time.**
- Instead of borrowing async-RL machinery (importance sampling, V-trace-style clips), AsyncOPD recomputes the reverse-KL signal under the *current* student policy at learner time. Rollout tokens are fixed but the weighting is redone.
- Empirically this beats the async-RL surrogates in the OPD setting — the paper argues those tools were designed for scalar rewards, not per-token KL.

**(3) Finite teacher-score caches → multi-sample MC.**
- Storing the teacher's full-vocabulary logits per token is prohibitive at frontier scale.
- Sparse / one-sample estimators are biased *or* high-variance; multi-sample Monte-Carlo estimators preserve MC correctability while cutting variance.

**Results.** 1.6× to 3.8× throughput over strict synchronous OPD at matching accuracy on the paper's benchmarks. Open-sourced.

## Why it matters

- **Makes OPD a first-class post-training tool.** Serial OPD was throughput-bounded; async unlocks the same scaling story as async RL.
- **Direction-of-KL as a systems knob.** The paper turns an abstract training choice into a concrete infrastructure decision — pick forward if you want deep pipelines, reverse if you can afford learner-time recomputation.
- **Bias-variance framing for teacher caches** generalises beyond OPD to any setting where a large auxiliary model provides per-token supervision.

## Gotchas & tricks

- **The recompute-at-learner-time trick still costs a student forward pass.** Cheaper than a fresh rollout but not free — budget accordingly.
- **Async-RL stabilisers don't help here.** Adapting them without testing can silently degrade OPD; use the paper's OPD-specific surrogate.
- **Multi-sample MC estimator hyperparameters are load-bearing.** Sample count controls the bias-variance floor; tune per model size.

## Sources

- Paper: *AsyncOPD: How Stale Can On-Policy Distillation Be?* — Kang et al. (FuriosaAI / Ajou / Berkeley / MSR / KRAFTON / Ludo Robotics), 2026 — [arXiv:2606.24143](https://arxiv.org/abs/2606.24143).
