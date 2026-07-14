# Trust Region Policy Distillation (TOP-D)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation (OPD) — training a student on its own rollouts scored by a teacher — is unstable and high-variance. TOP-D swaps the fixed teacher for a **dynamically-constructed proximal teacher** that stays close to the current student, giving a trust-region-shaped update with a formal monotonic-improvement bound. Zero extra compute vs baseline OPD.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [rlvr.md](./rlvr.md)

---

## What it is

**On-Policy Distillation (OPD)** trains a student by (1) sampling rollouts from the student itself, (2) scoring each rollout with a fixed, stronger teacher (e.g. teacher log-probs or teacher-assigned reward), and (3) updating the student to imitate the teacher-preferred behavior. It's the "distillation" analogue of on-policy RL: no off-policy correction, all trajectories come from the current policy.

OPD's problem is high variance. Early in training the student diverges from the teacher; the gradient signal from a distant teacher becomes noisy and often destabilizes training entirely.

**TOP-D** replaces the fixed teacher with a **proximal teacher** $\pi_{\text{teacher-prox}}$ constructed at each step to be within a KL ball of the current student. The proximal teacher itself is scheduled to move toward the true target teacher — small steps, always inside a trust region.

---

## How it works

### Proximal teacher construction

At step $t$, given the student $\pi_\theta$ and the true target teacher $\pi_{\text{teacher}}$, the proximal teacher is defined implicitly by:

$$
\pi_{\text{teacher-prox}}^{(t)} \in \arg\max_\pi \; \mathbb{E}_\pi[R] \quad \text{s.t.} \quad \mathrm{KL}(\pi \,\|\, \pi_\theta) \le \delta
$$

Intuitively: pick the best-scoring policy inside the KL ball around the current student. The paper derives a closed-form / regression-based construction (no separate optimization needed), so the proximal teacher is essentially free to compute from the true teacher's per-token scores and the student's log-probs.

### Distillation update

With the proximal teacher in hand, the student takes a standard OPD-style step against it:

$$
L_{\text{TOP-D}} = -\mathbb{E}_{o \sim \pi_\theta} \big[ A_{\text{prox}}(o) \cdot \log \pi_\theta(o) \big]
$$

where $A_{\text{prox}}$ is the advantage under the proximal teacher's implied reward. Because the proximal teacher is by construction close to the student, gradients stay well-scaled and variance is bounded.

### Scheduling

The proximal teacher's KL budget $\delta_t$ is scheduled — small at first, growing as training stabilizes. In the limit $\delta \to \infty$, TOP-D reduces to standard OPD; in the limit $\delta \to 0$, it reduces to no update. The paper shows that keeping $\delta$ in a moderate range preserves monotonic improvement while achieving fast convergence.

### Theory

TOP-D admits a **formal global convergence guarantee** and a **monotonic improvement bound**: under the proximal-teacher construction, each step is guaranteed to weakly improve the student's expected reward under the true teacher's implied objective. This is what "trust region" buys — the same guarantee TRPO buys for policy optimization.

---

## Why it matters

- **OPD becomes usable.** Prior OPD required aggressive learning-rate warmup and heavy regularization to avoid collapse. TOP-D removes the tuning problem via a construction-level fix.
- **Zero compute overhead.** No extra forward passes, no separate optimization step — the proximal teacher is derived from the same teacher-scoring pass OPD already does.
- **Distillation for reasoning models.** OPD is the natural recipe for shrinking an RL-trained long-CoT teacher into a cheaper student. Making it stable turns "distill R1 into a 7B student" from a research project into a reliable engineering step.
- **Bridge between SFT distillation and full RL.** SFT distillation caps at teacher capability with no exploration; full RL from scratch is expensive. TOP-D-style OPD sits in between with a stability guarantee.

---

## Gotchas & tricks

- **Proximal-teacher KL budget is the only real hyperparameter.** Too small: student barely moves. Too large: reverts to unstable OPD. The paper reports a schedule that grows $\delta$ over training.
- **Requires per-token teacher log-probs.** Like other on-policy distillation methods, TOP-D needs the teacher to score every student rollout — teacher inference dominates cost.
- **The monotonic bound is on the *proximal teacher's* objective**, not the true teacher's. In practice the two align, but if the proximal-teacher approximation is loose, guarantees weaken.
- **Advantage estimation reuses the OPD stack.** Any advantage estimator (per-response scalar, per-token, etc.) that works for OPD works for TOP-D — the change is only in the teacher policy, not the update rule.

---

## Sources

- Paper: *Trust Region Policy Distillation* — Xie, Zhang, Xie, Yang — [arXiv:2607.04751](https://arxiv.org/abs/2607.04751).
- Lineage: *Trust Region Policy Optimization* — Schulman et al., 2015 — the trust-region framing TOP-D borrows.
- Related: *On-Policy Distillation of Language Models* (baseline OPD papers) — the paradigm TOP-D stabilizes.
