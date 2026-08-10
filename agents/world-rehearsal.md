# World rehearsal (EnvACE)
*Depth — training tool-use agents by letting the policy play the environment.*

**TL;DR:** Agent RL usually needs an executable environment (real or simulated) to score tool calls. World rehearsal replaces external interaction with a single LLM that alternates two roles per turn — **actor** (issue a tool call) and **environment** (emit the tool's response) — with both roles trained end-to-end against the terminal task-success reward. The policy internalizes an environment model as a byproduct.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../agents/README.md](../agents/README.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

A training scheme (and inference trick) for long-horizon tool-use agents that removes the need for an external executable environment during training. The same weights that decide *what to do* also predict *what would happen if you did it*, and the two are optimized jointly under a task-success reward.

## How it works

**Per-turn structure during training:**

```
policy(state)   →  tool call a_t              # actor role
policy(a_t)     →  synthetic response o_t     # environment role
policy(state ∪ (a_t, o_t))  →  next tool call a_{t+1}
...
terminal reward R  →  applied to both roles in the same trajectory
```

Both roles share the base LLM's parameters; the trajectory contains both actor turns and environment turns, and the standard policy-gradient objective (GRPO in the paper) is computed over both. Task-success reward is terminal and RLVR-style (verifiable).

**At inference time:** the internalized environment model enables **private rehearsal** — the model runs one or more full internal rollouts of a candidate action before committing to it. A moderate rehearsal budget yields further gains without any real external call.

## Why it matters

- **Removes infra cost.** Executable environments are the biggest hidden cost of agent RL — every new domain needs sandboxing, tool wrappers, ground truth, and rate limits. World rehearsal amortizes that into weights.
- **Test-time compute lever.** Private rehearsal is a rejection-sampling-like scaling axis that costs no external calls.
- **New scaling axis.** "Internal world model" joins "long CoT" as an alternative to external tools/search — analogous to how internal reasoning eventually replaced external retrieval for some tasks.

Benchmarks: beats environment-scaling baselines on BFCL-v4, τ²-Bench, VitaBench, and FinMCP-Bench, with test-time rehearsal adding further gains.

## Gotchas & tricks

- The environment role must be trained end-to-end — freezing it or pretraining it separately loses the signal.
- Reward hacking risk shifts: the policy can implicitly reward its own hallucinated environment responses. Mitigation is keeping the terminal verifier *rule-based* (not learned).
- Rehearsal budget has diminishing returns; the paper reports "moderate" budgets are the sweet spot.
- Does not eliminate the need for an evaluation environment — you still need one to score final trained models.

## Sources

- Paper: *EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning* — Xu et al., 2026 — [arXiv:2608.06197](https://arxiv.org/abs/2608.06197)
