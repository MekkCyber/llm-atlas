# CoffeeBench

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A long-horizon **multi-agent economy** benchmark. Six LLM agents — two farmers, two roasters, two retailers — autonomously run businesses over a 90-day simulated coffee supply chain, each maximizing cumulative net income through communication, negotiation, and transactions while managing cash, inventory, and pricing. Unlike single-agent web/computer-use benchmarks, the environment **is** other LLM agents with private objectives.

**Prereqs:** none
**Related:** [gauntletbench](gauntletbench.md), [ia-bench](ia-bench.md), [../agents/README.md](../agents/README.md)

---

## What it is

Most agent benchmarks evaluate a single agent against a passive scripted environment, where "challenge" comes from task complexity. CoffeeBench inverts the setup: the environment **is** other LLM agents with their own goals and private state. To do well an agent must:

- form expectations about *other agents' strategies* and update those over time,
- communicate and negotiate prices through unstructured messages,
- maintain its own cash, inventory, and pricing across 90 simulated days,
- handle the absence of a single ground-truth optimal — equilibria are emergent.

Roles are heterogeneous — each tier (farm / roast / retail) has different cost structures, inventory dynamics, and downstream constraints, removing the symmetry crutch that simplifies symmetric multi-agent games like prisoner's-dilemma tournaments.

## How it works

1. **Six-agent setup.** Two of each role. Each agent runs a private LLM policy with access to the same per-turn observation (its own cash, inventory, recent messages, and price posts from counterparties).
2. **90-day horizon.** Each simulated day, agents act in role-order, exchanging messages, posting buy/sell offers, and accepting trades. End-of-day accounting updates cash and inventory.
3. **Heterogeneous environment.** Coffee production has a multi-stage supply chain with non-trivial constraints (perishable inventory at retail, time-to-roast at the roastery, weather-affected yields at the farm). The asymmetry forces real economic interaction.
4. **Evaluation.** Cumulative net income across the 90 days, per role and per agent. Also coarser system-level metrics (total welfare, price stability).

## Why it matters

- **Targets a capability gap that existing benchmarks miss.** Long-horizon multi-agent economic reasoning — coordination under incomplete information, price discovery, strategic negotiation — barely shows up in single-agent web or coding benchmarks.
- **Forces *real* memory and planning.** 90 simulated days mean an agent has to remember earlier transactions and counterparty behavior; short-context shortcuts fail.
- **Surfaces emergent failures.** Pricing collapse, oligopolistic collusion, and inventory deadlocks all appear as natural failure modes of weaker policies, giving the field concrete pathologies to study.
- **Complements existing benchmarks.** Where [GauntletBench](gauntletbench.md) probes visual / spatial agent shortfalls, CoffeeBench probes strategic / economic agent shortfalls. Both are evaluation-space gaps the existing graph lacks.

## Gotchas & tricks

- **Reward attribution is non-trivial.** A retailer's net income depends on the actions of roasters and farmers; a "smart" retailer paired with weak suppliers may still lose.
- **Specification gaming risk.** Agents that discover non-economic exploits (manipulating message-channel formatting, exploiting accounting precision) can score well without economic skill. Evaluators should monitor for these.
- **Frontier-model leaderboards are sensitive to per-agent prompt details.** Small system-prompt changes affect equilibrium behavior; report bootstrapped intervals over runs.
- **Symmetry breaks the easy strategies.** Two-of-each-role is just barely enough to avoid monopoly; benchmarks of this style with one-of-each role collapse into uninteresting equilibria.
- **Useful pair with [verification-horizon](../post-training/verification-horizon.md)** thinking — CoffeeBench is a natural place for "user-as-verifier" and "agent-verifier" reward constructions in agentic RL.

## Sources

- Paper: *CoffeeBench: Benchmarking Long-Horizon LLM Agents in Heterogeneous Multi-Agent Economies* — Hattori, Araragi, Ogawa, Onose, Makino, Usuki, Ishida, 2026 — [arXiv:2606.16613](https://arxiv.org/abs/2606.16613) — KPMG AZSA / Sakana AI.
- Background: *Multi-Agent Reinforcement Learning* surveys for general framing of heterogeneous multi-agent settings.
