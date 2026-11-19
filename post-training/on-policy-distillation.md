# On-Policy Distillation
*Depth — distill a teacher policy into a student using the student's own rollouts, with per-domain pitfalls.*

**TL;DR:** The student generates completions itself, the teacher scores each token, and the student is trained to match the teacher's distribution on those student-generated trajectories. Powerful for agent post-training and multi-teacher multimodal reasoning — but naively applied, it **reshapes agent behavior in domain-incompatible ways**.

**Prereqs:** [grpo](grpo.md), [_post-training](_post-training.md)
**Related:** [_rl](_rl.md), [rejection-sampling](rejection-sampling.md), [../multimodal/mpeg-diffusion](../multimodal/mpeg-diffusion.md)

---

## What it is

A middle ground between **SFT distillation** (student matches teacher on teacher's data) and **RL** (student rolls out and optimizes reward). In on-policy distillation:

1. The **student** generates the trajectory.
2. The **teacher** supplies a per-token target distribution over that same trajectory.
3. The student is trained with a KL (or reverse-KL) loss against the teacher's distribution on the student's own tokens.

The key asymmetry: training data is drawn from the *student's* distribution, not the teacher's — so the student learns to be the teacher at the places it actually visits, not the places the teacher visits.

## How it works

A step:

```
student rolls out  s_1, s_2, …, s_T
teacher scores:    p_T(s_t | s_<t)  for each t
loss:              KL( p_S(s_t | s_<t) || p_T(s_t | s_<t) )   per token, summed
```

Variants:

- **Single-teacher** (standard). One teacher, one student.
- **Multi-teacher (M3OPD-Reasoner).** One teacher per domain (image STEM, image perception, video reasoning). The teacher used for scoring is chosen per sample, by domain. Student distills across all three.
- **Agentic / ReAct (Kong et al.).** Teacher and student are ReAct agents (interleaved reasoning + tool calls). Distillation is applied over the token stream that includes both thought and action tokens.

**Position-selective distillation** (Kong et al.): keep the KL loss only at positions where the teacher is *strongly* confident. The student still generates freely but is only shaped where the teacher has a clear preference. Mitigates negative transfer in agent domains.

## Why it matters

- **Doesn't inherit teacher's blind spots.** Unlike SFT-on-teacher-data, the student visits its own states, so it learns a *corrective* fit — "when I visit position X, be like the teacher would at X."
- **Composes across teachers.** M3OPD-Reasoner's multi-teacher-into-one-student recipe works where naive multi-teacher SFT fails: each teacher only shapes the student on its own domain's trajectories, there's no cross-teacher interference.
- **Bridges SFT and RL.** Cheaper than full RL (no reward model, no exploration), stronger than SFT (student-aligned data distribution). Rapidly becoming default for agent post-training stages.

## Gotchas & tricks

- **Agent-domain negative transfer.** On-policy distillation can help one benchmark and hurt another at the same hyperparameters. Kong et al. trace this to teacher-induced ReAct-pattern shifts (think/act balance, step length, tool-call frequency). The student collapses toward the teacher's dominant pattern.
- **Position-selective mitigation is a knob, not a fix.** Keeping only teacher-confident positions reduces negative transfer but leaves some target-domain gain on the table.
- **Rollout budget dominates cost.** You need enough student rollouts per batch to cover its distribution; too few and the KL is noisy.
- **Reverse-KL is often preferable for peaky teachers.** Forward KL averages over teacher modes; reverse KL forces mode-seeking, which matches agentic trajectories better.

## Sources

- Paper: *M3OPD-Reasoner: An Open Recipe for Post-Training Multimodal Reasoning with Multi-Teacher On-Policy Distillation* — Phan, Sun, Zhang, Aw (VinUniversity / A*STAR), 2026 — [arXiv:2611.23034](https://arxiv.org/abs/2611.23034)
- Paper: *Agentic On-Policy Distillation Implicitly Reshapes LLMs' ReAct Patterns* — Kong, Li, Ren, Wei, Wei, Lian, Zhang, Ye, 2026 — [arXiv:2611.19244](https://arxiv.org/abs/2611.19244)
