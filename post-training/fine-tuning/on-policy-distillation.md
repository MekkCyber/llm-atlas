# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Knowledge distillation where the *student* generates the trajectories, and a stronger teacher scores those trajectories (or provides logits over them). Unlike offline distillation on a fixed teacher dataset, OPD keeps the training distribution matched to the student's current policy — a distillation-family answer to the same distribution-drift problem that RL fixes with on-policy rollouts. Recent work (**Flux-OPD**) extends OPD to open-ended domains by letting the teacher-conditioning **context evolve with the student**, using a decomposition of the reverse-KL objective to weight contextual corrections by an explicit conflict term.

**Prereqs:** [_post-training](../_post-training.md), [dpo](../dpo.md)
**Related:** [long2short](../reasoning/long2short.md), [_rl](../_rl.md), [rejection-sampling](../rejection-sampling.md)

---

## What it is

Offline distillation trains a student on a fixed dataset of teacher outputs — cheap, but the student never sees its own mistakes, so the training distribution drifts away from what the deployed model will actually generate. On-policy distillation flips this: **the student generates, the teacher supervises**. Concretely:

1. Sample a completion $y \sim \pi_\text{student}(\cdot \mid x)$ (on-policy rollout).
2. Score it with the teacher — either as logits (soft target) or as a reward (RL-flavored variant).
3. Update the student to move toward the teacher on that trajectory.

The reverse-KL objective $\mathbb{E}_{y \sim \pi_\text{student}} \bigl[\log \pi_\text{student}(y) - \log \pi_\text{teacher}(y)\bigr]$ is the canonical loss. OPD sits between two poles: **RLHF/RLVR** (rollouts + reward-scalar signal) and **offline KD** (teacher dataset + no rollouts).

## How it works

The vanilla loop, per step:

```
1. Sample x from prompt distribution
2. y  ← sample from π_student(·|x)
3. Compute reverse KL: L = Σ_t log π_student(y_t | y_<t, x) − log π_teacher(y_t | y_<t, x)
4. Backprop through student only
```

**Flux-OPD** ([Wang et al. 2026](https://arxiv.org/abs/2607.28022)) extends this for open-ended domains where there's no verifiable reward and contexts (system prompts, few-shot examples) are the primary way to convey preference:

- Decompose reverse KL into two terms:
  1. The student is distilled toward the **geometric mean of context-conditioned teachers**.
  2. A **conflict term** measures how much those teachers disagree.
- Treat `(context-conditioned teacher logits) − (context-free teacher logits)` as a **contextual correction signal**.
- Inject the correction into the context-free teacher anchor.
- **Weight the correction magnitude by the conflict term** — when teachers disagree strongly, trust the anchor more.
- Let the context *evolve with the student's performance*, so the target keeps yielding fresh supervision instead of collapsing into the student.

## Why it matters

- **Open-ended domains** (creative writing, ambiguous coding, chat quality) resist RLVR because there's no cheap verifier. OPD gives a distillation alternative that still sees the student's actual distribution.
- **Cheaper than RL.** No value network, no PPO ratios, no importance sampling weights. Just forward pass through teacher + reverse-KL loss.
- **Compositional with RL.** OPD and RLHF can share the same rollout infrastructure; the teacher just replaces the reward model.
- Flux-OPD's decomposition also **unifies context-based supervision** (a fundamentally SFT idea) with **on-policy training** (a fundamentally RL idea).

## Gotchas & tricks

- **Reverse KL is mode-seeking.** The student collapses onto whichever teacher mode is closest — you lose diversity. Use forward KL (or a mix) if that matters.
- **Static contexts stop teaching once distilled in.** This is the exact failure Flux-OPD's *evolving contexts* fixes.
- **Conflict weighting is essential** when multiple context-conditioned teachers give opposing signals — otherwise the student oscillates.
- Teacher inference cost scales with student rollout length; batch teacher calls or use a smaller teacher variant to keep throughput up.

## Sources

- Paper: *Flux-OPD: On-Policy Distillation with Evolving Contexts* — Wang et al., 2026 — [arXiv:2607.28022](https://arxiv.org/abs/2607.28022)
- Related: *β-OPSD* — Liu et al., 2026 — [arXiv:2607.28582](https://arxiv.org/abs/2607.28582) — analytic bridge between OPSD and policy optimization.
