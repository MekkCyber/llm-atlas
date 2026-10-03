# Strong-to-Weak LLM Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a smaller "student" LLM to imitate a stronger "teacher" by matching teacher-generated token distributions. Three design axes dominate practice: (a) **rollout policy** — rollouts come from the teacher (off-policy) or the student (on-policy); (b) **KL direction** — forward (student → teacher) or reverse (teacher → student); (c) **learning rate**. A 2026 systematic study shows no single preset dominates — the best choice depends on the objective, evaluation setting, and optimizer.

**Prereqs:** [../_post-training.md](../_post-training.md), [../rejection-sampling.md](../rejection-sampling.md)
**Related:** [../dpo.md](../dpo.md) · [../_rl.md](../_rl.md) · [../rlvr.md](../rlvr.md)

---

## What it is

Distillation is the backbone of small-model recipes (Qwen-Distill, R1-Distill, Phi, Gemma-small). The student learns from the teacher's *distribution over tokens* rather than only its argmax, which transfers more information than SFT on teacher-written text alone.

The target to minimise is a KL between student $\pi_s$ and teacher $\pi_t$:

- **Forward KL:** $\mathrm{KL}(\pi_t \| \pi_s)$ — covers the teacher's modes; student has to put mass everywhere the teacher does. Mode-covering; smooths out teacher noise.
- **Reverse KL:** $\mathrm{KL}(\pi_s \| \pi_t)$ — the student concentrates on *some* teacher modes; mode-seeking. Harder to optimise but can yield sharper students.

Rollouts can come from:

- **Off-policy (teacher rollouts).** The teacher writes the data; the student fits it. Cheap at inference time if the teacher data is pre-generated; close to standard SFT on teacher completions.
- **On-policy (student rollouts).** The student writes samples; the teacher scores them. More like RL — the loss is computed on distributions along the student's own trajectory.
- **Hybrid.** Mix of both.

## How it works

Standard distillation step:

1. Sample a prompt. Produce a response trajectory (teacher-written for off-policy; student-generated for on-policy; both for hybrid).
2. For each token position, compute the teacher's and student's distribution over the vocabulary.
3. Compute the per-token KL in the chosen direction and backpropagate through the student's parameters.
4. Repeat with the student's LR schedule.

The 2026 study (Piskorz, Berthon, van der Schaar) runs a controlled sweep across these axes on Llama-3 and Qwen-2.5 bases over scientific, medical and arithmetic reasoning tasks, and finds:

- **Forward KL is roughly stable across rollout policies.** On- and off-policy both work; the direction doesn't flip the ranking.
- **Reverse KL is sensitive to rollout policy.** On-policy helps reverse KL much more than it helps forward KL.
- **The best rollout policy is objective-dependent.** The common claim "on-policy is strictly better" doesn't hold once optimizer and benchmark vary.

## Why it matters

- **Right choice can swing accuracy several points** at the same compute budget.
- **Challenges the current default.** Many open recipes default to on-policy + reverse KL by analogy to RL; this is often suboptimal.
- **Separates distillation from RL conceptually.** Distillation's loss is a divergence to a *fixed* teacher distribution; RL's loss involves reward-weighted trajectories. Treating distillation as "softer RL" conflates the two.

## Gotchas & tricks

- **Reverse KL + off-policy** can collapse to over-concentrated students; pair with higher LR regularisation or add on-policy data.
- **Temperature on teacher.** Teachers sampled at $T=1$ expose their full distribution; a greedy teacher degrades to standard SFT on teacher argmax tokens. For continuous KL losses, keep teacher temperature > 0.
- **Logit truncation.** Teacher logits are huge tensors; many implementations top-$K$ truncate them to save bandwidth. The truncation reshapes the loss and can hurt forward KL more than reverse.
- **Student capacity.** If the student is far smaller than the teacher, forward KL (mode-covering) tends to be bland; reverse KL may give a sharper student at some capability cost.

## Sources

- Paper: *On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics* — Piskorz, Berthon, van der Schaar; University of Cambridge, 2026 — [arXiv:2609.35259](https://arxiv.org/abs/2609.35259). The current reference study for the three-axis sweep.
- Paper: *Distilling the Knowledge in a Neural Network* — Hinton, Vinyals, Dean, 2015 — original formulation (vision CNNs).
- Paper: *MiniLLM: Knowledge Distillation of Large Language Models* — Gu et al., 2023 — early LLM-scale reverse-KL distillation.
- Paper: *On-Policy Distillation of Language Models* — Agarwal et al., 2023 — introduced the on-policy framing.
