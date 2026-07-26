# Harness-Native Agent Training

*Depth — train the model inside the real inference harness it will be deployed with, not a hand-written proxy.*

**TL;DR:** Modern agents (Claude Code, Codex, OpenClaw) live inside elaborate inference harnesses that handle tool routing, memory, sub-agent spawning, and external system access. Open agent-training stacks have traditionally worked with hand-written toy environments, then paid a distribution-shift tax at deployment. Harness-native training frames the *harness itself* as the training environment: the RL loop sees whatever multi-turn flow the harness produces, gradients backprop through the model's calls, and evaluation and serving share the same execution graph. **OpenForgeRL** is the first published system built to this pattern.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../systems/ray.md](../systems/ray.md) · [experience-distillation.md](./experience-distillation.md)

---

## What it is

Traditional agent RL training loops look like this:

```
env = HandWrittenEnv()          # a Python class the researcher wrote
policy = LLM(model)
for step in RL_loop:
    obs = env.reset()
    while not done:
        action = policy(obs)
        obs, reward, done = env.step(action)
    update_policy(...)
```

But at deployment, the model doesn't see `HandWrittenEnv`. It sees Claude Code's tool router, its scratchpad memory, its sub-agent spawn logic, its permission dialogs, its context compression. The distribution shift can be arbitrary large — the same model that scored well against `HandWrittenEnv` can regress in the real harness for reasons that never appeared in training.

Harness-native training replaces `HandWrittenEnv` with the deployment harness itself, so the RL environment *is* the deployment environment.

## How it works

Three components:

1. **A harness adapter** that exposes the harness's step function (`(state, model_output) → (new_state, reward_signal, done)`) to the training loop, while the harness continues to run its own tool logic, memory, and multi-turn flow internally.
2. **Reward hooks** the harness can emit — success/failure signals, per-tool acknowledgements, or terminal outcome scores — that the RL loop consumes as reward.
3. **A rollout scheduler** that manages many harness instances in parallel, since one harness may hold real filesystem / shell state and cannot be trivially forked.

The model's calls happen through the harness's normal LLM interface; gradients flow back through those calls using standard RL post-training machinery (PPO, GRPO). The harness's *non-LLM* code (tool execution, memory, permissioning) is treated as an opaque environment step — no gradients through it.

## Why it matters

- **Closes the train / deploy distribution gap.** Every capability that the harness silently provides (retry logic, tool schemas, memory) is present at training time exactly as it will be at inference.
- **Enables research on production-style stacks.** Papers on agent RL have been constrained to whatever environments fit in a research codebase. A harness-native layer lets the same paper study the real deployment loop.
- **Standardises reward emission.** Once harnesses expose reward hooks, every downstream RL algorithm can consume them without reimplementing the environment adapter.
- **Right-sizes what the model must learn.** If the harness handles retries, the model doesn't need to learn to retry. If the harness compresses context, the model doesn't need to learn to be terse. Training against the same abstractions the harness provides means the model learns the residual — which is what should be RL-tuned in the first place.

## Gotchas & tricks

- **Harness statefulness constrains rollout parallelism.** A harness that mutates real files or hits real APIs cannot be forked arbitrarily. Snapshotting, containerisation, and rollback are prerequisites.
- **Reward wiring is bespoke.** Every harness exposes different signals. OpenForgeRL provides adapters for a few (CodeBuddy Code, Claude Code); adding a new harness is real integration work.
- **Reward hackability inside the harness.** The model may learn to game the harness's tool sequencing (e.g. call a permissive tool early to skip a check) rather than solve the task. Adversarial rollouts and outcome-only rewards defend against this.
- **Cost per rollout is much higher.** A real harness may spawn sub-agents, execute code, and hit external services — one rollout can cost 10–100× a hand-written env rollout. Sample efficiency (e.g. via Experience Distillation) becomes essential, not optional.

## Sources

- Paper: *OpenForgeRL: Train Harness-native Agents in Any Environment* — Yu, Peng, Xu, Zou, Wu, Cheng, Yao, Singh, Yu, Gao — Columbia / Dartmouth / Microsoft Research, 2026 — introduces the framework and harness adapters for CodeBuddy Code and Claude Code.
- Related: Ray for compute orchestration ([../systems/ray.md](../systems/ray.md)); RLVR framing for outcome-only rewards ([../post-training/rlvr.md](../post-training/rlvr.md)).
