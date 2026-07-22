# Tool-Call Boundary Calibration (Soft Clamp)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Multi-teacher on-policy distillation (one teacher good at tool calls, another good at direct answers) reliably improves tool-call recall — but silently shifts the student's decision boundary toward *over*-calling. The mismatch is invisible in aggregate KL: tool-call examples don't receive more token exposure or larger per-token divergence than direct-answer examples. Soft Clamp reframes the problem as **behavior-boundary calibration** and offers a localized training-time compression of extreme token-level Jensen-Shannon divergence that preserves nonzero gradients rather than hard-clipping.

**Prereqs:** [_rl.md](../post-training/_rl.md), [grpo.md](../post-training/grpo.md)
**Related:** [rejection-sampling.md](../post-training/rejection-sampling.md) · [_post-training.md](../post-training/_post-training.md)

---

## What it is

A training-time modification to multi-teacher generalized knowledge distillation (GKD) for tool-use agents. Sits inside the standard OPD loop; only the per-token loss-shaping changes.

## How it works

Standard multi-teacher GKD computes, per token, the JSD between the student and the (mixed) teacher distribution:

$$D_t = \mathrm{JSD}(\pi_\text{student}(\cdot\mid s_t) \| \pi_\text{teach}(\cdot\mid s_t))$$

Soft Clamp compresses $D_t$ at the tail: outlier tokens with unusually large JSD (which drive the boundary drift) are pulled toward a threshold rather than clipped hard. Concretely a smooth saturation $\tilde D_t = f(D_t; \tau)$ that leaves small divergences untouched, softly saturates above $\tau$, and — critically — preserves a nonzero gradient beyond $\tau$ so the model still learns from those tokens (unlike hard clipping, which zeros the gradient).

The paper compares four strategies on the same setup:

| Strategy | Where it acts | Cost |
| --- | --- | --- |
| **Hard clipping** | Training-time, per-token | Cheap; kills gradient on outliers |
| **Global reweighting** | Training-time, batch-level | Cheap; strong but non-local |
| **Soft Clamp** | Training-time, per-token | Cheap; local; preserves gradient |
| **Inference-time entry bias** | Deployment, decoding-time | Cheap; recovers frequency, not full profile |

## Why it matters

Multi-teacher distillation is how nearly every open-source agent gets its tool-calling capability — route different behaviours to specialists, distill down. The paper reframes what people had treated as "just a bit more over-calling" as a *behavior-boundary calibration* problem, and shows the fix is local: aggregate losses hide it, per-token localized shaping fixes it. On an APIGen-MT-derived decision set, Soft Clamp cuts over-calling from 14.2±2.1% to 9.0±0.2% while retaining most of the call-recall gain over SFT.

## Gotchas & tricks

- No strategy dominates across all metrics; Global Reweight is a strong non-local comparator and often within noise of Soft Clamp on aggregate scores.
- The tail threshold $\tau$ needs to be set relative to the run's own JSD distribution; a fixed absolute cutoff is brittle.
- Inference-time entry bias reproduces most of the *frequency* fix but only part of the joint (call-recall, non-tool-final, invalid-call) profile — training-time fixes are strictly stronger for downstream behavior.

## Sources

- Paper: *Diagnosing and Calibrating Tool-Call Boundary Drift in Multi-Teacher On-Policy Distillation* — Jiabin Shen, Guang Chen, Chengjun Mao (Ant Group), 2026 — [arXiv:2607.07050](https://arxiv.org/abs/2607.07050) · [HF](https://huggingface.co/papers/2607.07050)
