# NOPO: Negative-Only Policy Optimization
*Depth — RLVR variant for one-sided verifiers, in which the policy is updated only by suppressing verified failures.*

**TL;DR:** In many RLVR settings the verifier can confirm failures reliably but cannot confirm successes — unit tests catch bugs but passing tests may still hide shortcuts; math format-checkers catch format errors but not "right answer, wrong reasoning." **NOPO** treats passing responses as **unlabeled** and updates the policy *only* by suppressing verified failures, using a reference-anchored suppression loss plus a group-level adapter that rescales updates by the share of non-falsified responses.

**Prereqs:** [rlvr](rlvr.md), [grpo](grpo.md), [_rewards](_rewards.md)
**Related:** [ppo](ppo.md) · [rl-prompt-curation](rl-prompt-curation.md) · [../safety/reward-hacking.md](../safety/reward-hacking.md)

---

## What it is

RLVR assumes a verifier returns $+1$ for correct responses and $0$ for incorrect. NOPO is the variant for verifiers with **asymmetric reliability**: high precision on failure labels (a "failed" label is almost always right) but low precision on success labels (a "passed" label may hide bugs the verifier didn't catch). The standard "reward passes, suppress failures" GRPO update then drifts toward hackable passes — the policy gets rewarded for things that just happened to satisfy the verifier.

NOPO's cut: don't reward passes at all. Only *suppress* verified failures, and let the reference model anchor behavior on the rest.

## How it works

For each prompt, sample a group of responses $\{o_1, \dots, o_G\}$. Partition them with the one-sided verifier $V$:

$$
\mathcal{F} = \{ o_i : V(o_i) = \text{fail} \}, \qquad \mathcal{U} = \{ o_i : V(o_i) \ne \text{fail} \}
$$

Unlike GRPO, $\mathcal{U}$ is **unlabeled**, not positive. The loss has two terms:

1. **Reference-relative suppression** on $\mathcal{F}$:

$$
L_{\text{sup}} = \frac{1}{|\mathcal{F}|} \sum_{o_i \in \mathcal{F}} \left[\, \log \pi_\theta(o_i \mid q) - \log \pi_{\text{ref}}(o_i \mid q) \,\right]
$$

Minimizing $L_{\text{sup}}$ pushes the current policy's probability on verified failures *below* the reference model — i.e. unlearn them relative to a known-good anchor.

2. **Group-level adapter.** Scale the update by the share of non-falsified responses:

$$
\alpha_{\text{group}} = \frac{|\mathcal{U}|}{G}
$$

If almost every response in the group is falsified, the update is strong (the policy is clearly in bad territory). If only one is falsified, the update is small (the policy is near-converged on this prompt). This is NOPO's analogue of GRPO's group-normalized advantage.

The full objective is $\alpha_{\text{group}} \cdot L_{\text{sup}}$ plus a standard KL penalty to $\pi_{\text{ref}}$.

## Why it matters

- **Directly targets reward hacking.** The policy never gets positive gradient from a response the verifier merely *failed to disprove*, closing the single largest failure mode of RLVR.
- **Pareto gains on pass@k.** Across code, logic, and math benchmarks, the paper reports pass@k improvements over GRPO/RLVR baselines, holding under inference-time scaling and cross-benchmark transfer.
- **Drop-in for GRPO pipelines.** NOPO reuses the group-sampling structure of GRPO; only the loss and the label partition change. Existing rollout infrastructure ([../systems/partial-rollouts.md](../systems/partial-rollouts.md)) applies unchanged.

## Gotchas & tricks

- **The reference model does a lot of work.** $L_{\text{sup}}$ is anchored relative to $\pi_{\text{ref}}$ — if the reference is bad on the target distribution, NOPO's "suppression" becomes directionless. Keep $\pi_{\text{ref}}$ fresh (periodically update to the latest SFT checkpoint) rather than fixing it at the base model.
- **All-fail groups.** If $|\mathcal{U}| = 0$, $\alpha_{\text{group}} = 0$ and there's no update. This is the dual of GRPO's all-fail collapse (see [splice](splice.md) for an all-fail-group patch); NOPO's version is self-correcting but slow.
- **Not a free lunch for symmetric verifiers.** If your verifier is high-precision on both sides (e.g. a math checker that compares against a gold answer), GRPO's positive signal is real and NOPO leaves it on the table. Use NOPO only when success labels are genuinely untrustworthy.

## Sources

- Paper: *Negative-Only Policy Optimization for One-Sided Verifiable Rewards* — Xu, He, Zhang, Wang, An (NTU), 2026, arXiv 2611.17287.
- Related: [grpo](grpo.md) — base policy-optimization algorithm.
- Related: [../safety/reward-hacking.md](../safety/reward-hacking.md) — companion concept on how verifier failure modes drive reward-hacking policies.
