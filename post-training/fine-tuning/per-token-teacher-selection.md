# Per-Token Teacher Selection (MTA-OPSD)
*Depth — in on-policy self-distillation, let each token pick its preferred teacher from a pool, instead of using one fixed teacher per response.*

**TL;DR:** On-policy self-distillation (OPSD) typically uses a single teacher distribution per response. Tokens in a reasoning trace have different epistemic shapes — setup, work, finalization — and one teacher is the wrong aggregation. **MTA-OPSD** builds a **pool of teachers** that differ in reference truncation and exposure ratios; each token picks its preferred teacher via **majority voting** over top-$k$ candidates, then a **KL-based final selection**. On Qwen3 for mathematical reasoning, average gains up to **+5.9** points over the base model.

**Prereqs:** [README.md](README.md), [../_post-training.md](../_post-training.md)
**Related:** [../../fundamentals/z-loss.md](../../fundamentals/z-loss.md)

---

## What it is

On-policy self-distillation is a post-training variant of distillation where the "teacher" is a version of the model itself (e.g. a longer-rolled-out version, a higher-temperature version, or the model ahead of a few training steps). The student distribution $\pi_\theta$ is pushed toward the teacher's distribution $\pi_T$ on sampled trajectories via a KL term.

Standard OPSD uses one $\pi_T$ per response. MTA-OPSD argues this is the wrong aggregation level: different tokens benefit from different teachers, and committing to one teacher averages across a heterogeneous need.

---

## How it works

Three pieces:

1. **Teacher pool construction.** Build $M$ teacher distributions $\{\pi_{T_1}, \ldots, \pi_{T_M}\}$ from the same base model, differing along two axes:
   - **Reference truncation**: how much of the response context the teacher sees (full, half, prefix-only).
   - **Exposure ratio**: how much of the teacher's rollout comes from the student's policy vs. the teacher's own prior rollouts.

   Pool size $M$ in the paper is small (3–6).

2. **Per-token teacher selection.** At each token position $t$, run two stages:
   - **Majority voting over top-$k$ next tokens.** For each teacher $T_m$, take its top-$k$ predictions at position $t$. The teachers that agree on the mode (majority vote across the pool) form a candidate set.
   - **KL-based final selection.** From the candidate set, pick the teacher whose next-token distribution has **lowest KL** to the student's current distribution. This is the "closest useful teacher" for this token.

3. **Standard OPSD loss on the selected teacher.** Apply the KL loss $\text{KL}(\pi_{T_{m^*(t)}}(\cdot \mid x_{<t}) \,\|\, \pi_\theta(\cdot \mid x_{<t}))$ at each position $t$, where $m^*(t)$ is the selected teacher for that token.

Training proceeds like standard OPSD; the only change is the per-token selection step.

### Why the two-stage selection

- **Majority vote** filters out teachers whose predictions are outliers — a teacher that disagrees with the pool is probably wrong for this token.
- **KL to student** then picks among the agreeing teachers the one whose distribution is closest to what the student already knows. This gives a *small, learnable* correction rather than a far-away target that destabilizes training.

Taking either stage alone is weaker: KL-only converges to "whichever teacher happens to be closest" and can select the wrong one when teachers disagree; majority-vote-only can select a teacher far from the student's distribution, destabilizing the update.

---

## Why it matters

- **Per-token routing of training signal.** Converts a single-teacher problem into a routing problem, in the same spirit as [learned-model-router](../../agents/learned-model-router.md) but at training time and per-token granularity.
- **Composes with on-policy RL.** MTA-OPSD is applied as a term alongside RLVR or other policy-gradient losses; the paper reports additive gains.
- **Teachers are cheap.** All teachers are the base model under different conditioning — no extra pretraining, no new checkpoints.
- **Suggests a general pattern.** Many training procedures use one "target" distribution; this paper shows per-token routing between candidate targets is often a small code change with meaningful quality lift.

---

## Gotchas & tricks

- **Pool diversity matters.** If all teachers agree always, the selection is a no-op. Build teachers that differ meaningfully along reference-truncation and exposure axes.
- **$k$ in top-$k$ is a knob.** Too small and majority voting is unstable; too large and voting is uninformative. The paper uses $k=5$ as default.
- **KL-closest is still a bias.** Always picking the closest teacher can plateau training early — the student doesn't move much. Occasionally select stochastically among the top-voted candidates.
- **Per-token selection adds compute.** $M$ teacher forward passes per training step is $M\times$ cost over single-teacher OPSD. For small $M$ (3–4) this is acceptable; for larger pools, cache teacher logits.
- **Doesn't help when teachers are all weak.** The ceiling is still set by what the teacher pool knows collectively. If all teachers fail a token class (e.g. rare symbols), per-token selection can't fix it.
- **Only tested on reasoning so far.** The paper focuses on math; chat, code, and multimodal generalization are open.

---

## Sources

- Paper: *Every Token Needs Its Own Teacher: Multi-Teacher Adaptive On-Policy Self-Distillation* — Luo, Chen, Kasneci, Chen, Kasneci (TUM / MCML / relAI / ELLIS Institute Finland / Aalto), 2026, arXiv 2610.40574 — primary source.
