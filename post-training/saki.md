# SAKI
*Depth — route token-level OPD supervision by realized teacher accept / correct events, with maximal coupling giving an exact TV identity.*

**TL;DR:** SAKI (Wei et al., Meituan, 2026) runs a **KL-constrained teacher-guided rollout with maximal coupling** during on-policy distillation and routes token-level supervision based on realized agree/disagree events. Accepted positions keep sampled-token reverse-KL supervision; **correction positions receive direct supervision on the teacher's top token**. Under maximal coupling, the correction rate equals exactly $\mathrm{TV}(p_t, q_t)$ — so one trust-region radius bounds both rollout deviation and intervention frequency. An engine-resident speculative verifier gives a **4.22×** rollout throughput speedup.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md), [grpo.md](./grpo.md)
**Related:** [reasoning/long2short.md](./reasoning/long2short.md), [_rl.md](./_rl.md)

---

## What it is

Vanilla [on-policy distillation](./on-policy-distillation.md) applies the same KL-based supervision at every token the student samples. But when the student is weak, it visits prefixes far from the teacher's typical trajectories, and the teacher's per-token signal there is unreliable (correct in distribution, misleading at those states). SAKI replaces uniform supervision with **event-routed supervision**: at each token, record whether the teacher's and student's distributions produced the same symbol under maximal coupling, and supervise differently in each case.

## How it works

### KL-constrained teacher-guided rollout under maximal coupling

Let $p_t = \pi_{\text{teacher}}(\cdot \mid q, o_{<t})$ and $q_t = \pi_{\text{student}}(\cdot \mid q, o_{<t})$. Keep a KL-constrained hybrid proposal that stays within a trust region. Under **maximal coupling** of $p_t$ and $q_t$, the probability of a "correction event" (the two distributions disagree on the sampled token) is exactly:

$$
\Pr(\text{correct}) = \mathrm{TV}(p_t, q_t)
$$

So the same trust-region radius $r$ that bounds rollout drift also upper-bounds how often the teacher intervenes — a *single hyperparameter* governs both.

### Per-token supervision routing

- **Accept events** (student and teacher agree on this token under coupling): standard sampled-token reverse-KL supervision $\mathrm{KL}(q_t \,\|\, p_t)$.
- **Correction events** (disagreement): direct supervision on the teacher's **highest-probability token** (hard label), not the sampled one.

The routing is *conflict-adaptive*: supervision strength scales naturally with how often the student and teacher disagree.

### Engine-resident speculative verifier

To realize this at engine speed, SAKI exposes a speculative verifier inside the inference engine that preserves the exact-$q$ trajectory distribution and the coupling semantics while batching correction events. Reported matched-workload rollout throughput: **4.22×**.

## Why it matters

Across **7 mathematical reasoning benchmarks**, SAKI beats the matched teacher-guided baseline on **Mean@8 and Pass@8** for both **1.7B and 0.6B** students. Placement controls and fixed-prefix analysis confirm the correction-triggered routing (not a static schedule) is what carries the gains.

The deeper contribution is the identity $\Pr(\text{correct}) = \mathrm{TV}$ under coupling: it converts "how often is the teacher allowed to intervene" from a hand-tuned schedule into a principled, trust-region-controlled quantity. Portable to any method where a strong reference drives a weak proposer.

## Gotchas & tricks

- **Trust-region radius is now coupled.** You can no longer tune "how often does the teacher correct" independently from "how far does the student drift" — they're one knob. Nice for simplicity, less flexible.
- **Hard-label correction = argmax of teacher.** This can be too confident when the teacher distribution is flat; some runs benefit from top-k mixing on correction events.
- **Coupling requires same tokenizer.** Same-family OPD (student and teacher share a tokenizer) is the natural setting. Cross-family coupling is non-trivial.
- **Speculative verifier changes serving stack.** The 4.22× speedup needs engine-side support (SGLang / vLLM extension). Pure-training pipelines don't get the throughput win.

## Sources

- Paper: *SAKI: Maximal-Coupling-Routed Teacher Supervision for On-Policy Distillation* — Wei, Wang, Chen, Chai, Liu, Wang, Xu, Hu, Lin, Yin — Meituan, 2026 — [arXiv:2609.36601](https://arxiv.org/abs/2609.36601).
