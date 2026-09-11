# TITO — Training on Inference Token IDs

*Depth — train on the exact token IDs the sampler produced during rollout, with drift repair at turn boundaries, so the actor sees the same token stream the sampler saw.*

**TL;DR:** In long-horizon agentic RL, the tokens the *sampler* produced during rollout are not always the tokens the *training* pass sees — retokenization, tool-call insertion, and turn-boundary re-encoding all introduce silent drift. **TITO** (Training on Inference Token IDs) records the exact sampled token IDs from rollout and trains the actor against them, with explicit drift-repair logic at turn boundaries (where tool outputs get spliced in). Combined with Rollout Routing Replay it drops the train-inference log-probability gap from 0.021 to 0.013 on T1.

**Prereqs:** [rlvr](rlvr.md), [grpo](grpo.md)
**Related:** [rollout-routing-replay](rollout-routing-replay.md), [../fundamentals/bpe.md](../fundamentals/bpe.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

An agentic RL rollout produces a stream of tokens interleaved with tool calls, tool observations, and turn boundaries. The naive training loop re-tokenizes the resulting text before computing log-probs and gradients. Re-tokenization is *not* the identity:

- BPE-boundary drift: an original two-token span may re-tokenize to one token if the surrounding whitespace changes.
- Tool-output re-encoding: the observation string returned to the model may be normalized differently between rollout and training.
- Turn-boundary tokens: system/tool/user delimiters can re-align.

Each of these silently changes the token sequence the actor is being scored on. Over 300+-turn rollouts the drift accumulates and RL becomes unstable.

TITO removes the ambiguity: log the exact IDs the sampler emitted, train against them.

## How it works

**Rollout side.** The sampler already emits token IDs — persist them directly, not the decoded string. For every observation and tool result spliced into context, record the token-ID sequence used at that splice, exactly as the model saw it.

**Training side.** Reconstruct the training tensor from the recorded IDs, not from re-tokenizing the concatenated text. At each turn boundary, run a **drift-repair** step:

- Validate that the boundary tokens in the recorded stream still match the expected chat template.
- If a tool observation is spliced, insert the recorded observation tokens verbatim rather than re-encoding the string.
- Verify that no re-tokenization would collapse or split any recorded token pairs; if so, log and skip that trajectory rather than train on drifted tokens.

The loss is computed only over the *actor's* recorded tokens (not tool observations), same as usual.

## Why it matters

- **Cures a silent bug.** Retokenization drift is invisible in metrics until an unstable RL run reveals it; TITO removes the class entirely.
- **Enables clean turn-boundary handling.** Tool observations spliced verbatim keep the actor's context identical to what it acted on during rollout.
- **Necessary for very-long horizons.** Errors compound with turn count. T1 runs 300+ turns; without TITO the drift is intolerable.
- **Composes with R3.** [Rollout Routing Replay](rollout-routing-replay.md) closes the same drift at the routing layer. TITO closes it at the token layer. Both are needed.

## Gotchas & tricks

- **Cost is in storage, not compute.** Token IDs are cheap; log them alongside rewards and routing decisions.
- **Chat template must be stable.** If the training-side template changes between rollout and update, drift repair will reject many trajectories. Version the template.
- **Doesn't cover model-side drift.** Numerical/precision differences between rollout and training kernels are a *separate* source of divergence; TITO doesn't fix those.
- **Not the same as offline RL.** TITO is on-policy — the recorded rollout comes from *this* policy step. It's about consistency between sampling and gradient, not staleness.

## Sources

- Paper: *T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks* — Yang, Shi, Li, Wang, Li, Mi, Liang, 2026 — [arXiv:2609.11042](https://arxiv.org/abs/2609.11042).
