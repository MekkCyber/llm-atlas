# Hindsight Self-Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reward-free post-training recipe for models that jointly predict actions and future observations (world action models, trajectory generators). The *same* model is used as both student and teacher — but the teacher gets the demonstration's **actual future rollout** ("hindsight") as privileged input, and supervises the student on the trajectories the student itself generates from deployment-time inputs only. Distillation uses an **energy score** on actions. No environment, no reward model, no new data. Introduced in HINT-WAM for robot world action models.

**Prereqs:** [_rl.md](./_rl.md), [rejection-sampling.md](./rejection-sampling.md)
**Related:** [grpo.md](./grpo.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Policies that output both actions and visual futures can be improved without rewards by exploiting an asymmetry: a copy of the model that has *seen* the future finds it easy to produce a good action for the current state, while the real-deployment model has to predict without that help. Use the hindsight-informed copy to teach the deployment-mode copy, on exactly the states the deployment-mode copy visits.

---

## How it works

Let $\pi_\theta(a_t \mid s_t, o_{<t})$ be the student (standard deployment inputs) and $\pi^{\text{H}}_\theta(a_t \mid s_t, o_{<t}, o_{\geq t})$ be the teacher — same weights, extra input channel $o_{\geq t}$ (the actual future observation rollout from the demonstration).

Training loop:

1. **Sample trajectories from the student**: run $\pi_\theta$ in open-loop over demonstrations' initial states, collecting its own $o_{\geq t}$ (the student's *predicted* futures, from the generative branch).
2. **Query the teacher** $\pi^{\text{H}}_\theta$ on those same states, with the demonstration's actual future as privileged input.
3. **Distill** student → teacher action distribution using an **energy score**:
   $$
   \mathcal{L}_{\text{ES}} = \mathbb{E}_{a \sim \pi_\theta} \| a - a_H \| - \tfrac{1}{2} \mathbb{E}_{a, a' \sim \pi_\theta} \| a - a' \|
   $$
   which matches distributions rather than velocity fields — important when teacher and student share weights.

No simulator, no reward, no new data.

---

## Why it matters

- **Post-training without rewards.** For foundation-style robot policies (OpenVLA, RDT, π₀ family), reward engineering is the bottleneck. Hindsight self-distillation sidesteps it entirely.
- **Reduces forgetting.** Training on the student's own sampling distribution (not raw demo tokens) keeps pretrained skills intact on LIBERO-90 and RoboTwin 2.0.
- **General pattern.** Any model that generates both action and observation branches can be post-trained this way. The privileged-input-as-teacher trick likely transfers to video-generation policies, planning LLMs that emit both plans and world states, and any generative-world-model agent.

---

## Gotchas & tricks

- **Energy score over KL.** With weight-tied teacher and student, naive KL distillation collapses (student = teacher). Energy score keeps distributions from degenerating.
- **Student sampling matters.** Trajectories must come from the student (not the demonstration), or you're just doing BC. This is the "on-policy" ingredient without needing rewards.
- **Demonstrations provide the hindsight signal.** You still need demonstration data for the teacher's privileged input. HINT-WAM is "reward-free" but not "demonstration-free."
- **Mode of the teacher.** The teacher evaluates action quality *given the actual outcome*. If the demos themselves are suboptimal, the teacher inherits that ceiling.

---

## Sources

- Paper: *HINT-WAM: Hindsight-Informed Training for World Action Models* — Dat, Li, Wang (NUS) — 2026 — [arXiv:2611.24789](https://arxiv.org/abs/2611.24789).
- Background: *Energy scores* — Gneiting & Raftery, 2007 — proper scoring rule.
- Related: hindsight experience replay (Andrychowicz 2017) — same word, different mechanism (RL goal relabeling).
