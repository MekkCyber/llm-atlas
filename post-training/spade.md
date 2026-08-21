# SPADE
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** SPADE (Self-Play in Adaptive Synthetic Executable Environments) is a self-play RL framework in which a **single LLM plays two roles**: an *Environment Designer* that writes complete long-horizon training environments as executable code with a Gym-style `reset()`/`step()` interface, and a *Reasoning Agent* that learns to act inside them. The environment pool grows adaptively with the learner, breaking the fixed-goal-pool ceiling of hand-curated or statically-synthesized RL setups.

**Prereqs:** [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)
**Related:** [rl-prompt-curation.md](./rl-prompt-curation.md), [grpo.md](./grpo.md)

---

## What it is

An RL training loop where the *distribution of training environments* is generated online by the same model that acts in them. Because the designer's output is code implementing a standard Gym API, every synthesized environment is:

- **Executable** — you can literally `env.step()` on it.
- **Automatically verifiable** — the environment carries its own reward function, so no external verifier is needed. It composes naturally with RLVR.
- **Long-horizon by construction** — the designer is instructed to emit multi-step tasks rather than one-shot puzzles.

## How it works

1. **Designer role.** The LLM is prompted to emit a Python module implementing an OpenAI Gym-style env (`reset`, `step`, `observation_space`, `action_space`, terminal condition + reward). The prompt is conditioned on the agent's current skill profile.
2. **Executable check.** The candidate env is sandboxed and smoke-tested. Environments that don't run, always terminate immediately, or are trivially solvable are rejected.
3. **Agent role.** The same LLM (different context) plays the agent inside the accepted env. Rollouts produce trajectories with the env's own reward.
4. **RL update.** Standard policy-gradient RL (RLVR-shaped) using the env's reward signal. Both agent and designer heads update — the designer learns to emit envs the agent finds *learnable but not trivial*.
5. **Adaptation.** The distribution of accepted environments tracks the agent's frontier — as the agent improves, the designer proposes harder tasks.

## Why it matters

- **Attacks the "we're running out of RL tasks" bottleneck.** Existing agent-RL pipelines either hand-curate a few thousand tasks or use frozen synthetic verifiers. Both hit a ceiling. SPADE scales the environment pool with compute.
- **Verifier by construction.** Because envs are code, every environment ships with its own scorer — no separate reward-model training loop.
- **Gym-standard interface.** Any existing RL trainer that consumes Gym environments can plug into SPADE-generated envs.

## Gotchas & tricks

- **Reward hacking is the failure mode.** The designer can emit envs that trivially reward the agent (bag of degenerate wins). Mitigations: sandbox execution, difficulty scoring, and requiring the designer's own held-out policy to fail on the env before it's accepted.
- **Sandboxing is load-bearing.** Executing LLM-generated code demands strict resource, filesystem, and network isolation. A leaky sandbox will exfil-through-`step()` during training.
- **Curriculum tuning.** The designer's difficulty targeting knob is the whole game — too easy and the agent stagnates, too hard and rollouts return zero reward.
- **Not a substitute for domain envs.** SPADE synthesises task distributions in domains the base model can already imagine (programming, planning, text games). It does not conjure real-world robotic environments.

## Sources

- Paper: *SPADE: Self-Play in Adaptive Synthetic Executable Environments* — Liu et al., UW/CMU/MIT/Meta, 2026 — [arXiv 2608.19197](https://arxiv.org/abs/2608.19197) — introduces the self-play environment-generation framework.
