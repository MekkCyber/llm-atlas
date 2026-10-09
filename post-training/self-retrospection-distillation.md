# Self-Retrospection Distillation (SRD)

*Depth — distil a hindsight-aware pass of your own policy into its trajectory-blind foresight pass to recover gradient when all rollouts share the same reward.*

**TL;DR:** RLVR has a known blind spot: when every rollout in a group gets the same scalar reward (all correct, all incorrect), the group-relative advantage is zero and no gradient flows. SRD sidesteps the problem by running the **same policy** in two modes — a *hindsight* pass that sees the full completed trajectory, and a *foresight* pass that sees only the prompt — and distilling the hindsight prediction into the foresight. The hindsight copy is cheap (no extra training) and gives dense, trajectory-grounded signal exactly where RLVR's scalar reward falls silent.

**Prereqs:** [grpo](grpo.md), [rlvr](rlvr.md)
**Related:** [_distillation](_distillation.md) · [on-policy-distillation](on-policy-distillation.md) · [_rewards](_rewards.md) · [long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

Two modes of the same policy:

- **Hindsight** $\pi_\theta(\cdot \mid q, \tau)$ — conditioned on the prompt *and* the full rolled-out trajectory $\tau$.
- **Foresight** $\pi_\theta(\cdot \mid q)$ — conditioned only on the prompt (the "pre-interaction" distribution).

Hindsight's job is to predict what *should* have been anticipated. Foresight's job is to anticipate it in advance. SRD trains the foresight to imitate the hindsight on the same prompt:

$$
\mathcal{L}_{\text{SRD}} = \mathbb{E}_{q, \tau} \, D_{KL}\!\left( \pi_\theta(\cdot \mid q, \tau) \,\|\, \pi_\theta(\cdot \mid q) \right)
$$

The hindsight predictions can be token-level targets (choose action, plan step) or high-level intent summaries, depending on the task.

## How it works

Each training example yields a dense distillation signal even when the scalar reward is uninformative. For example, in a tool-integrated reasoning rollout:

- Scalar reward: 0/1 at the end — all 8 rollouts might score 0 → GRPO gradient = 0.
- SRD: for each of the 8 rollouts, run the hindsight pass and distil it into the foresight. The foresight update happens *per token*, not per rollout.

The hindsight model is literally the same weights as the foresight model — no second network, no extra training objective. The paper calls this "turning post-hoc experiences into prior foresight."

## Why it matters

- Breaks RLVR's **scalar-outcome floor**: when all rollouts tie, SRD still provides signal.
- Especially strong on **agentic tasks** where reward contrast is sparse (long trajectories, few successful ones).
- Reported gains of up to **+24.2 pp** across 10 tool-integrated reasoning and agentic benchmarks.
- Pairs well with [np-opd](np-opd.md) — anti-signal from a negative policy and dense hindsight signal are complementary.

## Gotchas & tricks

- **Hindsight is only cheap if the trajectory is already rolled out.** SRD stacks naturally on top of GRPO/RLVR training loops (reuse the rollouts).
- **Hindsight distribution ≠ true oracle.** The hindsight model can be wrong; the paper argues the KL loss implicitly averages out noise.
- **Watch for shortcut learning.** If hindsight reveals the final answer tokens, foresight can memorise them. Mask or paraphrase the reveal.
- **Length control.** Trajectories grow long on agentic tasks; standard long-context tricks apply to the hindsight pass.

## Sources

- *Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight* — Zhang, Chen, Hayashi, Li, Zhang, Zhang, Chen, Wu, Lu, Savarese, McAuley, Wu — Salesforce AI Research — 2026 — [arXiv:2610.08077](https://arxiv.org/abs/2610.08077)
