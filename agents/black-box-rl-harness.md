# Black-Box RL on Agent Harness
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** ClawGym II (Renmin U., 2026) trains agent policies against arbitrary, heterogeneous harnesses by treating the harness as a **black box** — the training loop can only step the environment, not introspect it. Combines **sandbox execution** (isolated per-rollout), **trajectory reconstruction** (rebuild coherent (state, action, reward) sequences from whatever the harness exposes), and **mix-harness training** (sample across many harness contracts per batch) to keep one policy general across many execution frames.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [_agent-harness.md](_agent-harness.md)
**Related:** [harness-scaling.md](harness-scaling.md) · [../post-training/grpo.md](../post-training/grpo.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Modern agent frameworks — ReAct loops, StateM-style runtimes, LangGraph, custom in-house harnesses — differ in what they expose, when they call the model, and how they represent state. If you train against one, the policy tends to overfit to its interface and break on the next. Black-box RL sidesteps that by refusing to depend on any harness internals: the trainer *sees only what the harness would give a human operator* — actions in, observations out, terminal reward at the end.

The result is a single training loop that can drive PPO/GRPO-style updates over a mixed batch of harnesses without per-harness plumbing.

## How it works

Three pieces stacked:

1. **Sandbox execution.** Each rollout runs inside an isolated sandbox — process, filesystem, network, and any external tool state contained. This means (i) rollouts don't cross-contaminate, (ii) a crash is contained, and (iii) the harness can be arbitrary user-provided code without threatening the trainer.

2. **Trajectory reconstruction.** Different harnesses log different things. The reconstruction step consumes whatever the harness happens to emit (stdout events, tool call logs, structured messages) and rebuilds a canonical `(observation, action, reward)` sequence the policy-gradient update expects. The policy sees a stable interface; the harness stays free to be weird.

3. **Mix-harness training.** Each RL batch samples rollouts across *multiple* harnesses simultaneously, so the gradient signal averages over harness-specific quirks. The policy learns behaviors that generalize rather than shortcut-optimize any one execution frame.

Combined, you can drop in a new harness by writing an adapter that emits enough log signal for reconstruction — no changes to the trainer.

## Why it matters

- Agent research is currently one large "harness fragmentation" problem: each lab ships its own framework and results don't transfer. Black-box training is one path to a policy that survives that fragmentation.
- The training-side complement to [harness-scaling](harness-scaling.md): StateM says the harness is where inference gains live; ClawGym II says the harness is also where training generalization lives.
- Sandbox execution is a hard prerequisite for scaling agentic RL to real tool use — without it, the trainer either has to trust harness code or run everything in a single degraded environment.

## Gotchas & tricks

- Trajectory reconstruction is where the leaky abstractions actually leak. If the harness silently drops observations, or emits actions out of order, the reconstructed sequence corresponds to no real rollout — the policy will chase phantom rewards. Validation of the reconstruction is not optional.
- Mix-harness training only helps if the harnesses actually disagree on interface. Sampling five harnesses that all look like ReAct in disguise buys nothing.
- Sandboxes cost real overhead per rollout (process startup, filesystem, network). Under-provisioning the sandbox pool becomes the RL bottleneck long before the model does.
- Black-box means no shaped rewards from harness internals — you get terminal signals. If the environment is very sparse, the mix-harness benefit isn't enough by itself; you'll need reward shaping or curriculum.

## Sources

- Paper: *ClawGym II: Exploring Black-Box RL on Agent Harness* — Huatong Song, Fei Bai, … Ji-Rong Wen — arXiv:2608.16798 — 2026 (Renmin University of China).
- Related: *StateM* (arXiv:2608.15089) — same "the harness is where the leverage is" thesis, on the inference side.
