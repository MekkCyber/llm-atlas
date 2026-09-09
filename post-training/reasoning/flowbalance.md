# FlowBalance
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A verifier-grounded self-improvement loss for reasoning models. On the student's own rollouts, a frozen "privileged" copy of the same policy produces token-level self-guidance scores; the group-relative verifier advantage (GRPO-style) then *gates the sign* of that guidance — retained on positive-advantage trajectories, reversed on negative ones, disabled when the rollout group has no outcome preference. Fit as a trajectory-balance objective; no separate imitation loss.

**Prereqs:** [grpo](../grpo.md), [rlvr](../rlvr.md), [long-cot-rl](./long-cot-rl.md)
**Related:** [../on-policy-distillation](../on-policy-distillation.md), [long2short](./long2short.md), [orm](./orm.md), [prm](./prm.md)

---

## What it is

A single-loop replacement for pure-RLVR reasoning training that fuses two supervision channels — a sparse verifier reward and a dense same-model self-guidance — while making them **agree** rather than compete. The self-guidance comes from the same policy conditioned on privileged context (a reference, a plan) that the student does not have at inference; the verifier decides whether to trust that guidance for each rollout group.

## How it works

For a batch of prompts $q$ and $G$ rollouts $\{o_i\}$ per prompt:

1. **Verifier advantage.** Score each rollout, compute group-relative advantage $\hat A_i$ (GRPO): $\hat A_i = (r_i - \mathrm{mean}) / \mathrm{std}$.
2. **Self-guidance score.** A frozen training-time view $\pi_T$ of the same policy — conditioned on privileged information — produces per-token log-probability gains $\Delta_t = \log \pi_T - \log \pi_\theta$ on the student's tokens. Sum to a trajectory score $S_i = \sum_t \Delta_t$.
3. **Outcome-calibrated fusion.** Combine into an energy:
   - $\hat A_i > 0$: keep $S_i$ (guidance reinforces successful trajectories).
   - $\hat A_i < 0$: **reverse** $S_i$ (a "false-positive self-guidance" correction — the model shouldn't be pushed toward tokens the same-model teacher liked on a failed rollout).
   - Group is all-positive or all-negative: guidance disabled, verifier alone drives the update.
4. **Trajectory balance fit.** The resulting energy exponentially reweights a reference policy. Fit with a profiled trajectory-balance objective — one log-partition estimate per rollout group, no separate token-level imitation loss.

Theoretical properties the paper establishes: within-group contrast preservation, a minimum-change reverse-KL characterization, monotonic verifier control of target reward, and an exact correction against false-positive self-guidance on rejected responses.

## Why it matters

- **Composes dense and sparse signals correctly.** Prior methods either apply dense guidance blindly (OPSD → collapse) or discard it (pure RLVR → sparse signal). FlowBalance uses the verifier as a *sign gate* over the dense self-guidance so the two never fight.
- **Mode-collapse resistant.** On AIME24, correct-strategy diversity is higher than pure OPSD and FlowRL; response-length collapse absent.
- **Beats FlowRL on Qwen3-4B and Qwen3-8B math, with faster and more stable training.**

## Gotchas & tricks

- **Privileged information must be honest.** The self-guidance signal only makes sense if the training-time view has access to something the inference-time student doesn't (reference solution, plan, verified environment feedback). Feeding leaky "privileged" info that overlaps with the eval distribution reintroduces contamination.
- **All-positive / all-negative groups are common on curriculum extremes.** In those groups the guidance drops out and training reduces to plain RLVR — this is by design, not a bug.
- **One log-partition per group.** The trajectory-balance formulation avoids per-token imitation losses but does need a decent group size $G$ to estimate the partition; too small $G$ raises variance.

## Sources

- Paper: *FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience* — Huang, Panaganti, Mi, Liang, 2026 — [arXiv:2609.03241](https://arxiv.org/abs/2609.03241).
- Related: [grpo](../grpo.md) (source of the group-relative advantage used as the verifier signal), [../on-policy-distillation](../on-policy-distillation.md) (the OPSD-adjacent family FlowBalance improves on).
