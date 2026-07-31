# StealthBench — Operational Stealth for Offensive-Security Agents

*Depth — a benchmark that separates "task solved" from "task solved without giving yourself away".*

**TL;DR:** Autonomous offensive-security agents can find real vulnerabilities but frequently leave loud OPSEC fingerprints — embedding credentials in public uploads, deleting production resources to prove access, force-adding uninvolved users to demonstrate a race condition. StealthBench measures this. 14 dockerized scenarios derived from 11 hand-verified real-world bug-bounty / red-team incidents, scored by a 3-LLM judge panel across six OPSEC dimensions with **safe success rate** (solved AND stealthy), **Stealth@Solve** (tradecraft quality among solves), and **reckless solve rate** (solved but cover blown). No model exceeds 54% safe success.

**Prereqs:** [../safety/_attacks.md](../safety/_attacks.md), [../agents/README.md](../agents/README.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../safety/prompt-injection.md](../safety/prompt-injection.md), [README.md](./README.md)

---

## What it is

A benchmark for autonomous offensive-security agents that decouples **task success** from **operational stealth** — the discipline of achieving an objective without revealing your presence, capabilities, or collected intelligence. The observation: agents that find real vulnerabilities but blow their cover are dangerous, not competent.

## How it works

**Task construction.** 11 OPSEC incidents were extracted from real bug-bounty and red-team engagement trajectories. Each represents a case where a human operator found a vulnerability but committed a stealth failure the field's tradecraft would consider inexcusable. These are expanded into **14 dockerized scenarios** — reproducible environments where an agent is asked to solve the same task class.

**Six OPSEC dimensions** cover the tradecraft space:

- Credential hygiene (don't leak found secrets).
- Non-destructive proof (don't delete prod to prove access).
- Minimal impact (don't drag in uninvolved users / accounts / resources).
- Trace hygiene (don't leave discoverable artifacts).
- Timing / rate discipline (don't trigger IDS with a scan storm).
- Attribution hygiene (don't tag your work with an identifiable signature).

**Scoring.** A 3-LLM judge panel scores each dimension with majority vote. Three headline metrics:
- **Safe Success Rate** — solved *and* stealthy across all dimensions.
- **Stealth@Solve** — tradecraft quality among successful solves.
- **Reckless Solve Rate** — solved but cover blown.

## Why it matters

- **Reframes offensive-security agent capability.** "Did it find the bug" is only half the question; "would you actually let it run in production" needs the stealth axis too.
- **Systematic OPSEC failure across model families.** No evaluated model exceeds 54% safe success — the failure mode is structural, not per-model.
- **Reusable "safe success" pattern.** Other domains have the same shape: medical (correct answer + no misinformation), legal (right advice + no disclosure), operations (task done + no side effects). StealthBench's split-metric structure ports directly.
- **Public leaderboard + eval harness.** Enables both defender research and OPSEC monitoring of deployed offensive-security agents.

## Gotchas & tricks

- **Judge panel disagreement.** Six-dimension OPSEC scoring across three judges yields some non-trivial variance; the majority-vote aggregation smooths but doesn't eliminate it. Read confidence intervals, not point estimates.
- **Dual-use disclosure.** Publishing "how to embed credentials in public uploads" is exactly the kind of trace real attackers already know. StealthBench doesn't teach new attacks; it teaches evaluators to catch known bad patterns.
- **Sample size is modest.** 14 scenarios from 11 incidents — small by ML-benchmark standards. Treat rankings as signal, not verdict.
- **Doesn't measure defensive tradecraft.** Purely on offensive-agent OPSEC. Blue-team-agent evaluation is a distinct (and important) gap.
- **Reckless solves are the specific alarm.** A model with high Reckless Solve Rate is worse than a model that just fails — it's an agent that will accomplish the task *loudly* if deployed.

## Sources

- Paper: *StealthBench: Measuring Operational Stealth in Autonomous Offensive-Security Agents* — Wood, 2026. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
- Site: https://stealthbench.com/ (leaderboard + eval harness + dataset).
