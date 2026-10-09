# On-Policy Distillation (OPD)

*Depth — a distillation recipe where the student generates the rollouts and a frozen teacher scores them token-by-token.*

**TL;DR:** The student samples a response, then the teacher's per-token distribution is used as the training target (typically forward KL from student to teacher). Unlike SFT distillation (fixed teacher outputs), OPD exposes the student to its *own* mistakes while still receiving dense per-token supervision. Powerful but **fragile**: when the teacher's token preferences misalign with downstream quality, OPD reward-hacks into long, repetitive generations.

**Prereqs:** [_rl](_rl.md), [grpo](grpo.md)
**Related:** [_distillation](_distillation.md) · [np-opd](np-opd.md) · [self-retrospection-distillation](self-retrospection-distillation.md) · [rejection-sampling](rejection-sampling.md)

---

## What it is

Three ingredients:

1. **Student policy** $\pi_\theta$ — the model being trained.
2. **Teacher** $\pi_T$ — a frozen, stronger model.
3. **A loss** that pulls the student's token distribution toward the teacher's **on the student's own rollouts**.

The canonical loss is forward KL on each token position:

$$
\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{s \sim \pi_\theta} \sum_t D_{KL}\!\left( \pi_T(\cdot \mid s_{<t}) \,\|\, \pi_\theta(\cdot \mid s_{<t}) \right)
$$

The student sees sequences it would actually generate. The teacher provides a per-token target distribution, not just a scalar.

## How it works

Reframed as RL (Cui et al. 2026): the teacher is an **implicit reward model**. Each token the student samples gets an implicit reward equal to the teacher's log-probability of that token. The policy gradient becomes:

$$
\nabla_\theta J \approx \mathbb{E}_{s \sim \pi_\theta} \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_{<t}) \cdot \log \pi_T(a_t \mid s_{<t})
$$

This is a policy-gradient update with the teacher's log-probability as the reward signal. It explains two empirical facts:

- OPD cannot create capabilities the student can't already sample (RL's exploration ceiling).
- When teacher log-probs correlate poorly with quality (long, hedging, repetitive tokens score high), the student **reward-hacks** that signal: outputs balloon in length, loop, and lose semantic quality.

### Mitigations (both standard)

- **SFT initialisation.** Start from a model whose output distribution already overlaps the teacher's; shrinks the KL and the exploration gap.
- **Mask unhealthy rollouts.** During training, drop rollouts with runaway length or repetition before computing the loss.

## Why it matters

OPD is the current workhorse for small-model recipes: Minitron, Qwen-small, Gemma-distill all use variants of it. The RL framing in Cui et al. clarifies when it works (reliable teacher preferences) and when it collapses (long-tail quality divergence). It is the base technique on top of which [np-opd](np-opd.md) and [self-retrospection-distillation](self-retrospection-distillation.md) build.

## Gotchas & tricks

- **Collapse mode is length-first.** Monitor average output length during training; a drift upward is the earliest signal.
- **Teacher evaluation ≠ teacher generation.** The RL framing says: pick teachers whose *scoring* is reliable, not whose *generation* is strong. The two don't always coincide.
- **Forward vs. reverse KL.** Forward KL is standard (mass-covering); reverse KL (mode-seeking) is sometimes used to avoid averaging over teacher modes.
- **Temperature.** Student sampling temperature changes the exploration-vs-stability tradeoff. Teacher logits are usually used at $T = 1$.

## Sources

- *Gains and Collapse in On-Policy Distillation: A Reinforcement Learning Perspective* — Cui, Yan, Luo, Zhang, Fu, Zhang — 2026 — [arXiv:2610.03185](https://arxiv.org/abs/2610.03185)
