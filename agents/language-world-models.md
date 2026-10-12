# Agentic Language World Models (Trace2Env)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When the real environment is unavailable, use another *agent* as the environment: a world-model agent that consults a trace-derived "worldbook" plus persistent episodic state to predict each action's observation and lasting side effects. Training-free. On nine environments, improves next-observation fidelity and long-horizon consistency over prompt-based language world models.

**Prereqs:** [README.md](./README.md), [../data/README.md](../data/README.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

A pattern for standing up an environment replica from historical interaction traces alone, without reconstructing the backend. The replica is an LLM agent (the "world-model agent") that reads structured memories and simulates observations.

## How it works

Two artefacts and one loop:

1. **Environment worldbook.** Offline, distill agent-environment traces into a structured book indexed by state and action patterns. Entries record what observation a given state+action combination produced historically, with provenance.
2. **Persistent episodic state.** During a simulated episode, maintain a running symbolic/natural-language state object capturing durable side effects (files written, items taken, flags flipped).
3. **Simulation loop.** At each step, the task agent emits an action; the world-model agent consults the worldbook with the current state+action, updates the episodic state, and returns the predicted next observation.

No base-model training is required — the whole thing runs on prompting + the pre-built worldbook + the episodic state.

## Why it matters

Training agents against proprietary or defunct APIs usually requires reconstructing the environment, which is expensive or impossible. Agentic language world modelling drops that requirement: if you have traces, you can get a replica. The 9-environment evaluation shows it beats pure prompt-based world modelling on both next-step fidelity *and* long-horizon consistency, the two numbers that matter for RL training.

## Gotchas & tricks

- Worldbook quality is a direct ceiling. Thin or biased traces produce a thin or biased replica.
- Persistent episodic state is where consistency lives — missing a side effect once can cascade into incoherent trajectories.
- The replica is only as good as the LLM running the world-model agent at *predicting* observations. Factual errors in the worldbook propagate.
- Pattern-match prone: states that don't match any worldbook entry fall back to generative hallucination; worldbook coverage audits are necessary before using the replica for RL.

## Sources

- Paper: *From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation* — Chen et al., HK PolyU / NTU, 2026 — [arXiv:2610.06100](https://arxiv.org/abs/2610.06100)
