# Language World Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A language world model is an LLM trained to predict the next *environment state* given an action and prior observations — terminal output, file diff, tool response, GUI screenshot description. It serves as a high-fidelity simulator that replaces expensive real-environment rollouts for agentic RL, and doubles as a strong agent foundation model when warm-started this way. Qwen-AgentWorld (Qwen Team, 2026) is the first large-scale instantiation, with 35B-A3B and 397B-A17B MoE variants covering seven domains.

**Prereqs:** [README.md](README.md), [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../post-training/grpo.md](../post-training/grpo.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) · [../case-studies/qwen-agentworld.md](../case-studies/qwen-agentworld.md)

---

## What it is

Classical world models in RL learn a transition function `p(s' | s, a)` over compact state vectors so a policy can plan in imagination instead of in the environment. The bottleneck for agent learning has been that environments are expensive (rollouts of terminal sessions, browser actions, OSWorld trajectories) and unstable (servers fail, APIs throttle).

A language world model is the same idea at LLM altitude: the state is a *textual* environment description and the transition is an LLM call. Given a prompt of the form

```
<env-description> + <action history> + <next action>
```

the model predicts the next environment description in natural language. The simulator is then just LLM inference — cheap, batchable, deterministic enough for RL.

## How it works

The training pipeline reported in Qwen-AgentWorld has three stages:

| Stage | Data | Objective |
| --- | --- | --- |
| Continued pretraining (CPT) | 10M+ real interaction trajectories across 7 domains + augmented professional corpora | Next-token prediction over state transitions; injects generic world-model priors |
| SFT | Curated next-state prediction with long chain-of-thought | Activates explicit reasoning over what should happen next |
| RL | Tailored framework with hybrid rubric+rule rewards | Sharpens simulation *fidelity* — predicted states must match ground-truth executions |

Two deployment modes from one model:

- **Decoupled simulator.** Use the language world model in place of the real environment during agentic RL. The agent rolls out actions; the simulator returns predicted state transitions; the agent's policy updates against environment-grounded rewards. Rollout cost drops by orders of magnitude vs. real environments.
- **Unified agent foundation.** Use the world-model checkpoint as a warm-start for downstream agent SFT/RL. The world-modeling objective doubles as broadly-useful agent pretraining.

## Why it matters

- Agentic RL has been gated by environment cost; if a learned simulator preserves enough fidelity to give honest reward signal, the RL stack runs entirely in inference clusters.
- World-model-warm-started agents outperform agents trained directly, on the same benchmarks — suggesting the world-model objective is also a strong general agent pretraining task.
- Cross-domain coverage (terminals, OS, tools, browsers, …) means a single simulator can replace many bespoke ones.

## Gotchas & tricks

- Simulator fidelity is the whole game. Errors compound across long rollouts; reward signal from a low-fidelity simulator misleads the policy. The hybrid rubric+rule reward (see [../post-training/hybrid-rubric-rule-reward.md](../post-training/hybrid-rubric-rule-reward.md)) is the lever for keeping fidelity high.
- Domains differ in predictability. Deterministic substrates (shells, tools) simulate cleanly; stochastic ones (browsers, network APIs) need explicit randomness or accept some divergence.
- The simulator is itself an LLM: serving costs at scale rival the policy's. Practical deployments share inference between the agent and its world model.
- Open question: does world-model pretraining suffice for *all* downstream agent tasks, or only the ones inside the 7-domain training distribution? Qwen-AgentWorld's numbers are promising but cover only the trained domains.

## Sources

- Paper: *Language World Models for General Agents* (Qwen-AgentWorld) — Qwen Team, 2026 — [arXiv:2606.24597](https://arxiv.org/abs/2606.24597).
- Benchmark: AgentWorldBench (Tool Decathlon, Terminal-Bench 1.0/2.0, OSWorld-Verified, etc. — rubric-graded across 5 dimensions).
- Reference (classical world models): *World Models* — Ha & Schmidhuber, 2018.
