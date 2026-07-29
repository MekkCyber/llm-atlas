# On-Policy Agentic Distillation
*Depth — post-training regime that shapes multi-turn long-horizon planning by matching student rollouts to teacher(s) on-policy.*

**TL;DR:** Post-training regime for agent LLMs in which the *student generates the trajectory* (multi-turn tool-use rollouts in a controlled environment) and one or more *teachers* supply the supervision signal for each step. Contrasts with off-policy trace SFT (student never explores) and pure outcome RL (no per-step teacher). The multi-teacher variant blends supervision from several strong policies and stabilizes long-horizon planning where single-teacher distillation collapses.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../agents/README.md](../agents/README.md)

---

## What it is

Multi-turn long-horizon planning ability is acquired through several training stages (pre-training, SFT, RL), but disentangling their contributions on Internet-trained models is hard. On-policy agentic distillation is a *post-training* regime that isolates the planning-improvement signal: the student explores the environment on-policy, and one or more teacher policies attach step-level supervision (per-turn preferred action, per-turn reasoning target, or KL toward the teacher's action distribution).

## How it works

- **Controlled environment.** A multi-turn task simulator where task length, data quality, planning knowledge, and planning patterns can be varied independently.
- **Student rollouts.** The current policy generates multi-turn trajectories in that environment.
- **Teacher(s) score / target.** One teacher (single-teacher) or several teachers (multi-teacher) provide per-step targets — usually a next-action distribution the student is regularized toward, plus optional outcome rewards.
- **Loss.** A weighted combination of (a) KL from student action distribution to teacher's on the student's own state and (b) outcome reward on completion. Multi-teacher blends targets before or after the KL.

## Why it matters

Two things fall out of the controlled setup: the field can now *attribute* planning ability to specific training stages (pre-training vs SFT vs on-policy distillation), and multi-teacher on-policy distillation stabilizes long-horizon planning across task lengths where single-teacher distillation collapses. This is a template other groups can reuse when their agent RL runs turn brittle at long horizons.

## Gotchas & tricks

- Off-policy SFT on teacher traces gets stuck at teacher behavior — no exploration means no way to correct student-specific failure modes.
- Single-teacher on-policy distillation inherits the teacher's blind spots; if the teacher is weak on a subclass of states the student sees often, the loss collapses to zero on the wrong distribution.
- Multi-teacher mixing helps *only* if teachers actually disagree in useful ways; identical teachers add nothing.
- Reward-signal timing matters — sparse outcome-only rewards at long horizons still under-supervise even with a teacher; you generally want teacher-KL at every turn.

## Sources

- Paper: *The Physics of Multi-Turn Long-Horizon Planning: From Pre-training to Post-training via Single- and Multi-Teacher On-Policy Agentic Distillation* — Men et al., 2026 — [arXiv:2607.24720](https://arxiv.org/abs/2607.24720)
