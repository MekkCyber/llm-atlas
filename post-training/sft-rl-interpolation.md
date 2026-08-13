# SFT ⇄ RL Checkpoint Interpolation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** After RL post-training, don't ship the RL checkpoint directly — **linearly interpolate** its weights with the pre-RL SFT checkpoint and ship the interpolation. Cheap, portable, and consistently recovers general capability that pure RL runs partially lose, without giving up the RL objective's gains on the target task. Used as the shipping recipe for **MiLMMT-46-v1.0** (Xiaomi, 2026); a small-scale, single-axis cousin of model soups.

**Prereqs:** [grpo.md](grpo.md), [../pre-training/model-souping.md](../pre-training/model-souping.md)
**Related:** [_rl.md](_rl.md), [rlvr.md](rlvr.md)

---

## What it is

RL post-training on top of an SFT checkpoint (with GRPO, PPO, DPO, or similar) reliably improves the target metric but often causes drift on unrelated capabilities:

- Multilingual coverage narrows toward high-reward languages.
- Instruction-following on out-of-distribution formats regresses.
- Broader knowledge queries get shorter or more generic.

Ways to fight this drift:

- **Tune the KL penalty $\beta$** up. Slows the drift but also slows the gains.
- **Mix general-domain rollouts** into RL. Adds engineering burden and can dilute the RL signal on the target.
- **Interpolate SFT and RL checkpoints.** Cheap post-hoc mixing that trades off drift and target gain with a single scalar.

The interpolation approach is what MiLMMT-46-v1.0 ships. Concretely, for models $\theta_\text{SFT}$ and $\theta_\text{RL}$ trained from the same base:

$$
\theta_\text{final} = (1 - \alpha) \cdot \theta_\text{SFT} + \alpha \cdot \theta_\text{RL}, \quad \alpha \in [0, 1].
$$

Choose $\alpha$ on a held-out mix of target-task and general-capability evals.

## How it works

Two properties make this work:

- **Linear-mode connectivity for near-neighbor checkpoints.** When $\theta_\text{RL}$ is initialized from $\theta_\text{SFT}$ and RL doesn't drift the weights too far, the linear path between them stays in a low-loss basin. This is the same regime that makes model soups work.
- **The RL and SFT loss landscapes disagree on unrelated capabilities.** The interpolation lets you dial in "how much RL" you want while keeping the SFT stability floor.

Practical recipe:

1. Train the base with SFT → save $\theta_\text{SFT}$.
2. RL-post-train from $\theta_\text{SFT}$ → save $\theta_\text{RL}$.
3. Sweep $\alpha \in \{0.3, 0.5, 0.7, 0.9\}$ on a small held-out set that includes both the RL target metric and general-capability probes.
4. Ship the $\alpha$ that jointly maximizes both.

## Why it matters

- **Zero extra training cost.** You already have both checkpoints. No new rollouts, no new losses.
- **Portable.** Works with any RL algorithm (GRPO, PPO, DPO) as long as SFT and RL share a base.
- **Beats reasonable alternatives on the frontier.** In the MiLMMT-46 study, SFT⇄RL interpolation *reached and matched* the quality frontier achieved by on-policy distillation — a much more expensive alternative — and beat both endpoints on their combined eval.

## Gotchas & tricks

- **Requires a shared base.** SFT and RL checkpoints must be trained from the same base weights; independent runs from scratch break linear-mode connectivity.
- **Optimizer-state divergence is not an issue.** Only the parameters are interpolated. Optimizer state is discarded.
- **Sweep $\alpha$ per-layer isn't necessary.** A single global $\alpha$ tends to work as well as per-layer sweeps in this small-drift regime; save the engineering time.
- **Not a substitute for KL regularization during RL.** If RL drifts too far, the interpolation basin breaks. Keep $\beta$ sane during RL.
- **This is a *single-pair* soup.** Full model soups average many checkpoints from a hyperparameter sweep. SFT⇄RL interpolation is the minimal, most common instance.

## Sources

- Paper: *Reference-Free Post-Training of Open Large Language Models for Multilingual Machine Translation* — Han, Gao, Fu, Luan (Xiaomi), arXiv 2608.10812, 2026. Uses SFT⇄RL interpolation as the shipping recipe for MiLMMT-46-v1.0.
- Related: model soups — Wortsman et al., 2022 — the general "average many fine-tuned checkpoints" idea. See [model-souping.md](../pre-training/model-souping.md).
