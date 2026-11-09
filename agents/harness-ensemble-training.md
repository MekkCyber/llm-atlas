# Harness-ensemble training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agent RL results are famously harness-dependent: ReAct, reflection loops, and planner-executor scaffolds each fix different failure modes, and no single harness is universally best. Harness-ensemble training keeps a fixed pool of specialized *sibling* harnesses, invokes them only on tasks the base harness completely fails, and uses those rescued trajectories as training data for a policy that is still evaluated under the base harness. Introduced in Harness-S (2611.04277).

**Prereqs:** [rl-prompt-curation](../post-training/rl-prompt-curation.md), [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [partial-rollouts](../systems/partial-rollouts.md)

---

## What it is

A **data-generation step** for agent RL / SFT. The policy is trained to succeed *under one fixed harness* (the evaluation harness), but the training trajectories are drawn from a pool of harness variants, biased toward the ones that happen to rescue specific failures.

The method decomposes "which scaffold should we use?" into two decisions:
- **Train-time:** which sibling harness has the best shot at rescuing this failed task?
- **Test-time:** fixed — the original harness, always.

## How it works

Setup: base harness $H_0$, sibling pool $\{H_1, \ldots, H_k\}$ (hand-designed variants: more tool-call scratchpad, explicit planning phase, self-reflection, etc.).

1. **Filter.** Run the current policy under $H_0$ on the training distribution. Collect tasks where $H_0$ completely fails (0% success over $n$ rollouts).
2. **Route.** For each failed task, score siblings by (a) behavior-profile similarity between task and a sibling's success distribution, and (b) LLM voting (an LLM rates "which sibling best fits this task?").
3. **Rescue.** Run the top-ranked sibling and keep any successful trajectory.
4. **Train.** Use rescued trajectories for SFT or as positive-reward samples in RL. The policy sees sibling-generated behavior but never the sibling harness at inference.
5. Iterate.

## Why it matters

- **Decouples scaffold from weights.** The policy learns to recover without a harness switch, which eliminates the "which prompt wrapper did the authors use?" bakeoff lottery.
- **Zero deployment cost.** Siblings exist only during training; the serving stack ships $H_0$ unchanged.
- **Diversity as a data lever.** Many scaffolds act as a diversity source for rollout sampling, cheaper than adding more models.

Reported gains: consistent improvements over policy-only training on AppWorld, SWE-Gym, DBBench.

## Gotchas & tricks

- The router matters: a bad router routes every task to the same sibling and the ensemble collapses. The LLM-voting signal is the main defense.
- Keep the sibling pool **small and interpretable** — each sibling should address a named failure mode.
- Rescued trajectories are off-policy with respect to $H_0$; SFT handles this cleanly, but RL needs importance-correction or a KL anchor to $H_0$.
- Risk: the policy learns behaviors that only make sense under a sibling harness and fail to translate — verify on held-out tasks under $H_0$ only.

## Sources

- Paper: Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training — Jiao et al., 2026 — [arXiv:2611.04277](https://arxiv.org/abs/2611.04277)
