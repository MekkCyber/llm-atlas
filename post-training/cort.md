# CoRT — Counterfactual Replay for Token-Level Rubric-Guided GRPO

*Depth — a token-level credit weighting for rubric-conditioned GRPO, without an auxiliary scorer.*

**TL;DR:** Standard GRPO broadcasts a single response-level advantage to every token. When rewards come from a *rubric* with several criteria that target different spans (formatting, factual span, style), the flat broadcast wastes signal. CoRT rescores the sampled response twice — once under the original rubric-conditioned prompt and once under a matched criteria-free prompt — and uses the per-token log-likelihood contrast as a proxy for how much each token depends on the rubric context. Those contrasts become bounded, response-normalized weights that redistribute the signed GRPO advantage across tokens. No auxiliary scorer, no separate relevance-learning stage.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [_rewards.md](./_rewards.md)
**Related:** [ppo.md](./ppo.md), [reasoning/prm.md](./reasoning/prm.md), [dpo.md](./dpo.md)

---

## What it is

A refinement of [GRPO](./grpo.md) for the **rubric-based RL** setting: the reward is not a scalar but the aggregation of several criterion-level judgments. GRPO reduces the aggregate to a single response-level advantage $A_i$ and broadcasts it to every token in response $o_i$. CoRT keeps the aggregation intact but replaces the flat broadcast with a **per-token weight** derived from the policy itself.

## How it works

For each sampled response $o_i$ from prompt $q$ with rubric context $R$:

1. **Two log-likelihood passes** on the *same* response:
   - Rubric-conditioned: $\ell^R_{i,t} = \log \pi_\theta(o_{i,t} \mid q, R, o_{i,<t})$
   - Criteria-free: $\ell^\varnothing_{i,t} = \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})$
2. **Per-token contrast**: $\delta_{i,t} = \ell^R_{i,t} - \ell^\varnothing_{i,t}$ — positive where the token is easier to predict *given* the rubric, i.e. the token depends on rubric content.
3. **Bound + normalize**: map $\delta_{i,t}$ through a bounded function (e.g. clipped tanh) and normalize across the response so the weights average to 1. This preserves the total advantage — CoRT redistributes credit but doesn't inflate it.
4. **Reweighted GRPO update**: replace the flat $A_i$ with $A_i \cdot w_{i,t}$ inside PPO's clipped ratio; everything else in [GRPO](./grpo.md) is unchanged.

No new networks, no reward-model modifications, no per-token PRM training. The signal comes from the policy's own likelihood under a counterfactual prompt.

## Why it matters

- **Free credit assignment.** GRPO's flat advantage is known to waste signal on rubric-style rewards; the usual fix is a learned token-scorer (PRM-style) that costs a separate training stage. CoRT gets a token-level signal from two forward passes on an already-sampled response.
- **Stability of GRPO preserved.** Because weights are bounded and response-normalized, the update stays inside GRPO's stable regime; no reward-scale re-tuning required.
- **Composes with rubric aggregators.** The rubric itself can still combine criteria however you like (sum, max, learned); CoRT operates strictly at the credit-redistribution layer.

Reported result: +4.4 pp average over response-level GRPO across instruction-tuned models and reward granularities; competitive with learned token-level baselines despite adding no scorer.

## Gotchas & tricks

- **The criteria-free prompt matters.** If the "matched" prompt still hints at the rubric (leaked criteria, boilerplate structure), $\delta_{i,t}$ collapses to zero and CoRT reduces to plain GRPO. Match the prompt shape carefully.
- **Two forward passes per response.** Rollout cost is unchanged, but training-time forward cost roughly doubles for responses (backward pass still runs once). Cheap compared to training a separate scorer.
- **Weights must be bounded.** Unbounded log-likelihood contrasts can spike on rare tokens and destabilize the PPO update. Clipped tanh / winsorization is the standard fix.
- **Interacts with KL regularization.** Because CoRT changes *where* credit lands within a response, the KL-to-reference penalty tunes differently — expect to re-visit the KL coefficient.
- **No benefit when the rubric is one-dimensional.** If every criterion measures the whole response uniformly (e.g. a single "is this correct" judge), $\delta_{i,t}$ carries no useful structure. CoRT shines when different criteria target different spans.

## Sources

- Paper: *CoRT: Counterfactual Replay for Token-Level Rubric-Guided Policy Optimization* — Zhang et al., Nanjing U. consortium, 2026 — introduces the method. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
- Related: [GRPO](./grpo.md) (the base algorithm), [PRM](./reasoning/prm.md) (the learned-scorer alternative CoRT competes with).
