# Multi-Teacher On-Policy Distillation (Road MOPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Multi-domain OPD normally routes each domain to its own teacher, but all domains share the student's parameters. When teachers disagree sharply, gradients fight and the "losing" domain degrades. Road MOPD treats the *non-routed* teachers as cross-domain critics, uses agreement-guided token-level credit assignment, and projects out gradient components that would hurt the disadvantaged teacher's domain — with an adaptive norm floor so the step doesn't vanish.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md), [grpo.md](./grpo.md)
**Related:** [_distillation.md](./_distillation.md) · [rejection-sampling.md](./rejection-sampling.md) · [_rl.md](./_rl.md)

---

## What it is

In multi-domain post-training (math + code + chat + safety, say), the standard "mixture of experts of teachers" is to pick a teacher per prompt by domain and run OPD against that teacher. The problem: the student is one network. When the routed teacher's gradient conflicts with what the other teachers would prefer, nothing watches the losses — the step still goes, and the disadvantaged domain quietly drops.

Road MOPD makes the non-routed teachers vote on each student rollout and uses that vote in two places: **token-level credit** and **gradient projection**.

## How it works

Per student rollout $(q, o)$ with routed teacher $T_r$ and other teachers $\{T_j\}$:

### Token-level credit via agreement

Each non-routed teacher $T_j$ scores (or logit-matches) the rollout. Define a per-token agreement signal $a_t$ summarizing how often the non-routed teachers endorse the token $o_t$. Scale the per-token advantage from the routed teacher by $a_t$: tokens the other teachers also like get full weight; tokens only the routed teacher likes get down-weighted.

Intuitively: if the math teacher wants a token but the code and safety teachers hate it, the step still happens, but with reduced magnitude.

### Preventive gradient projection

Even weighted, the step can still have components in directions that damage non-routed domains. Compute reference gradients $g_j$ for each non-routed teacher $T_j$. Project the current update $g_r$ to remove components anti-aligned with any $g_j$ (negative dot product):

$$g_r' = g_r - \sum_{j: \langle g_r, g_j \rangle < 0} \frac{\langle g_r, g_j \rangle}{\|g_j\|^2} g_j.$$

Pure projection can collapse the step to zero when conflicts are high. The **adaptive norm-retention floor** rescales $g_r'$ back toward $\|g_r\|$ by a factor that depends on how much projection removed — so the step stays non-trivial even in high-conflict regimes.

## Why it matters

- **Names the failure mode.** Multi-teacher OPD silently loses capability on the disadvantaged domain unless something catches cross-domain conflict. Road MOPD makes this a measurable, controllable signal.
- **Reuses teachers you already have.** No new models — the non-routed teachers are the same ones routed on other prompts. Compute increases roughly linearly in teacher count, not quadratically.
- **The ideas generalize.** Agreement-weighted credit and gradient projection with norm retention apply to any multi-signal post-training stack (e.g. multiple RMs, verifier + preference RM in R1-style Stage 4).

## Gotchas & tricks

- **Compute scales with the number of teachers.** Scoring $K$ teachers on every rollout costs $K\times$ the single-teacher scoring. Keep $K$ small (3 is the sweet spot in the paper).
- **Agreement signal is noisy.** Low-signal rollouts (very easy or very hard prompts) produce near-uniform agreement; the credit weights saturate and the method degenerates to vanilla OPD.
- **Projection direction matters.** Project against *negative* dot products only. Projecting against positive ones kills alignment you want.
- **Norm floor needs tuning.** Too high and the projection has no effect; too low and you reproduce the collapse problem. The paper uses an adaptive schedule driven by observed conflict magnitude.
- **Not a substitute for data balancing.** If domains are hugely unbalanced, Road MOPD can only partially compensate — the underlying data mixture still matters.

## Sources

- Paper: *Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation* — Tian, Liu, Zhu, Bai, Yang, Liu, Yu, Yang, Zhou, Li, Sun, Simiu, 2026 — [arXiv:2610.13521](https://arxiv.org/abs/2610.13521). Reports >5-point gains in high-conflict settings over domain-routed MOPD baselines.
