# Token-Share Balancing
*Depth — equalizing per-domain token contribution to the gradient update in multi-domain training.*

**TL;DR:** In multi-domain RL post-training and multi-teacher distillation, long-form domains dominate the token budget by construction (proofs and code are longer than instruction-following outputs), starving concise domains of gradient signal. Token-share balancing reweights per-domain contributions to the loss so each domain's share of the effective gradient matches its target share of capability — decoupled from raw token count.

**Prereqs:** [rlvr.md](rlvr.md), [multi-teacher-on-policy-distillation.md](multi-teacher-on-policy-distillation.md)
**Related:** [_rl.md](_rl.md)

---

## What it is

When several domains share a training batch, the *natural* per-domain gradient weight is proportional to the number of tokens each domain contributes — which is not the same as their contribution to capability. Long-output domains (math derivations, code) get a bigger share of the gradient than short-output domains (instruction-following, format-constrained answers) even when the intended mix is balanced.

Token-share balancing rescales per-domain losses so each domain's effective contribution matches a **target share** independent of raw token count.

## How it works

Given per-domain losses $L_1, \ldots, L_K$ contributing token counts $n_1, \ldots, n_K$, the vanilla per-batch loss is $L = \sum_i n_i L_i / \sum_i n_i$ — weighted by token count. Token-share balancing replaces the weights with a target-share vector $s_1, \ldots, s_K$ (e.g. uniform $1/K$):

$$
L_\text{balanced} = \sum_i s_i \cdot \tilde{L}_i, \qquad \tilde{L}_i = \frac{1}{n_i} \sum_{t \in \text{domain}_i} L_t
$$

Concretely: normalize each domain's token-level loss by its own token count first (turning it into a per-domain mean loss), then combine domains under the target-share weights. Long-form domains no longer dominate purely by output length.

In Open-MOPD (Gao et al., 2026), token-share balancing is one of three orthogonal levers that together lift headroom recovery from 35.6% to 83.4% in a controlled multi-teacher on-policy distillation benchmark.

## Why it matters

Concise-task collapse (instruction-following degrading during multi-domain training) is a widespread but under-discussed failure mode. It looks like a subtle capability regression but is really an artifact of the batch loss weighting all tokens equally when some domains produce 10x as many per prompt. Token-share balancing is a small, general fix that decouples target capability share from output length — applicable to any multi-domain RL or distillation setup, not just M-OPD.

## Gotchas & tricks

- **Not the same as per-example loss normalization.** Per-example normalization equalizes long and short *examples*; token-share balancing equalizes long and short *domains*. Both can be applied together.
- **Target shares need justification.** Uniform target shares assume all domains matter equally; if the deployment target has a real usage mix, use that as the target vector instead.
- **Interacts with dynamic budget allocation.** Static token-share balancing sets the floor; dynamic budget allocation on top of it (Open-MOPD's second lever) then routes extra budget to domains still far from ceiling.

## Sources

- Paper: *Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation* — Gao et al., 2026 — [arXiv:2608.19098](https://arxiv.org/abs/2608.19098)
