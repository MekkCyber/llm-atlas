# Harness-Policy Optimization (Self-HPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Coding-agent training that alternates two loops: (1) the **policy** (model) proposes revisions to the **harness** (scaffolding — prompts, tool wiring, control flow, retry logic) from its own failure traces, and (2) RL updates the policy under the chosen harness. Hard training tasks become the search set for the next round. Pushes Qwen3.5-35B-A3B's SWE-Bench resolve rate from 57.8% → 64.8% averaged across three bench variants, with gains that transfer out-of-distribution.

**Prereqs:** [grpo](../post-training/grpo.md), [rlvr](../post-training/rlvr.md)
**Related:** [rl-prompt-curation](../post-training/rl-prompt-curation.md), [rejection-sampling](../post-training/rejection-sampling.md)

---

## What it is

In production, a coding agent's success rate depends on two very different artefacts: the *weights* (what the model knows) and the *harness* (how the scaffold routes tool calls, retries, parses outputs, and recovers from errors). These are usually optimised separately — harness by hand, weights by RL — and co-tuning them has been empirically hard.

Self-HPO closes the loop. Instead of fixing the harness and training the policy, the *policy itself* is used to propose harness revisions based on failure traces it generated. The RL loop then trains the policy under the chosen harness. On the next outer iteration, failures from this round feed back into both. Importantly, the harness is a *first-class trainable artefact*, not infrastructure.

## How it works

One outer iteration of Self-HPO:

1. **Rollout under current harness.** The current policy $\pi_\theta$ runs on the task pool through the current harness $H_t$. Collect failure traces.
2. **Propose harness candidates.** $\pi_\theta$ (or optionally a stronger external model) is prompted with failure traces and asked to produce candidate harness revisions — new tool wrappers, modified routing logic, extra verification steps, prompt edits.
3. **Score candidates.** Each candidate runs on a held-out subset of the task pool. Keep the top-$m$ by solve rate.
4. **Policy RL under the best candidate.** Standard group-relative RL (GRPO-style) updates $\pi_\theta$ against the solve-reward under the chosen harness $H_{t+1}$.
5. **Hard-task curation.** Tasks the policy still fails become the search set for the next outer round.

Because the harness is proposed *by the policy* (not a separate meta-learner), the loop is fully bootstrapped — no external harness-authoring model is required, though the paper shows plugging in a stronger external proposer (Opus 4.8 class) further widens the gap.

## Why it matters

- **The harness is trainable.** This result makes it concrete that the scaffold around a coding agent is a search space, not infrastructure. Hand-authored harnesses leave points on the table.
- **Shared training signal.** Failure traces drive both the harness search and the RL curriculum. Previously these were separate pipelines with separate signal pools.
- **OOD transfer holds.** The gains persist on SWE-Bench variants not used for training, suggesting the harness improvements aren't memorising task-specific hacks.
- **Pairs with inference-time harness search.** Models like HarnessPro (also 2026) do Pareto harness search at inference. Self-HPO plus HarnessPro = train-time co-optimisation plus deploy-time budget-aware search.

## Gotchas & tricks

- **Harness proposals must execute.** A material fraction of policy-proposed harnesses throw errors on first run; filter with a cheap syntax/semantics check before scoring.
- **Reward hacking shows up as harness over-engineering.** The policy will add elaborate retry logic that technically improves solve rate on the training set but destabilises on OOD. Rate-limit harness complexity per round.
- **GRPO group size matters.** Group-relative advantages need enough rollouts per task to be low-variance; the paper uses 8+ rollouts per task.
- **Pair with hard-task curation, not easy.** Easy-task curation collapses the harness search; the whole point is to steer harness revisions at the policy's current failure modes.
- **Stronger external proposer compounds.** Letting Opus 4.8 propose harnesses (while the small policy does the RL) delivers the strongest numbers — a cheap way to inject capability from a larger model.

## Sources

- Paper: *Self-Evolving Coding Agents via Harness-Policy Optimization* — Wei, Yao, Peng, Wu, Cheng, Gao, Wang, Meng (MSR + CMU + UVa), 2026 — [arXiv:2610.22791](https://arxiv.org/abs/2610.22791).
- Background: [grpo](../post-training/grpo.md), [rlvr](../post-training/rlvr.md), [rl-prompt-curation](../post-training/rl-prompt-curation.md).
