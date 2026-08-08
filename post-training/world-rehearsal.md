# World Rehearsal (EnvACE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An agentic RL recipe that replaces external environment interaction with **the policy itself playing both agent and environment**. On even turns it emits a tool call; on odd turns it plays the environment and generates the tool response; then it conditions on that rehearsed response for the next decision. Both roles are jointly optimized end-to-end against the task reward. The model ends up carrying an implicit world model in its weights.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [rl-prompt-curation.md](./rl-prompt-curation.md).
**Related:** [rlvr.md](./rlvr.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [../agents/README.md](../agents/README.md) · [rejection-sampling.md](./rejection-sampling.md)

---

## What it is

Agent RL is bottlenecked by *executable environments*: dockerized web sandboxes, tool proxies, or synthesized simulators. They're expensive to build, brittle to maintain, and hard to scale. World Rehearsal trains the agent to *hallucinate* its own environment responses during training, then optimizes both roles jointly so the hallucinations converge to correct dynamics under the reward signal.

## How it works

**Alternating role assignment.** Format the trajectory as:

```
<sys> system prompt </sys>
<user> task </user>
<action_1>   ...        </action_1>   ← agent role
<env_1>      ...        </env_1>      ← environment role (also produced by model)
<action_2>   ...        </action_2>
<env_2>      ...        </env_2>
...
<final>      answer     </final>
```

Both `<action_k>` and `<env_k>` are emitted by the same policy. During training, the outcome-level reward (task success) backprops through the whole sequence, updating both roles simultaneously.

**Two consistency pressures shape the environment role.**

1. Task reward: if the rehearsed environment is inconsistent (says a search returned nothing when it would have returned a match), the resulting action leads to failure and the environment role gets a negative gradient.
2. (Optional) grounding: mix in some real environment traces to anchor the rehearsals.

**Test-time private rehearsal.** After training, the agent can rehearse candidate actions internally *before* committing an external tool call — a cheap deliberation step that adds an inference-time budget knob rather than an execution-time one.

## Why it matters

- **Decouples agent-RL scaling from environment engineering.** Training becomes pure policy compute; devops is no longer the pacing constraint.
- **Grounds a "world model" inside the LLM's weights** without a separate architecture — reusable at inference time.
- Outperforms environment-scaling baselines on BFCL-v4, τ²-Bench, VitaBench, and FinMCP-Bench, with further gains from test-time rehearsal.

## Gotchas & tricks

- **Reward hacking via cooperative hallucination** is the central risk: the environment role can lie in ways the action role rewards, e.g. always report success. Mitigations: mix in real traces, keep the reward outcome-grounded (verifier on final answer, not on intermediate tool responses), and hold out at least some tools with real execution.
- **Doesn't help when tool responses carry information the model can't recover** (fresh data, stochastic APIs). World rehearsal works because tool outcomes are largely predictable from context.
- **Curriculum matters.** Starting from a supervised warmup where environment turns are real tool outputs gives the environment role a head start before it has to hallucinate.
- **Inference cost balloons** if you leave `<env_k>` generation on at deployment. Route real tools for production and reserve rehearsal for training and optional test-time deliberation.

## Sources

- Paper: *EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning* — Xu et al., SJTU/ZJU/Tencent, 2026 — [arXiv:2608.06197](https://arxiv.org/abs/2608.06197).
- Code: https://github.com/Within-yao/EnvACE
