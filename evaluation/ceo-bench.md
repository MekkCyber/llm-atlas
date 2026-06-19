# CEO-Bench

*Depth — long-horizon agent benchmark: operate a simulated startup for 500 days.*

**TL;DR:** Most agent benchmarks test isolated short-horizon tasks (SWE-Bench, customer service). CEO-Bench bundles the four skills that real long-horizon work requires — navigating long horizons under uncertainty, acquiring information in noisy environments, adapting to a changing world, and orchestrating many moving parts — into a single scenario: run a startup for 500 simulated days. Success metric: final cash above the $1M starting balance. Only Claude Opus 4.8 and GPT-5.5 finish above; neither consistently turns a profit.

**Prereqs:** —
**Related:** [README](README.md), [aime](aime.md)

---

## What it is

A programmatically-graded simulator benchmark for long-horizon agentic capability. Not a question-answering benchmark, not a single-task workflow — a 500-day open-ended operation where the agent makes hundreds of interdependent decisions and survives or fails based on cumulative outcomes.

## How it works

**Setup.** The agent is dropped into a simulated startup environment with $1M starting capital, a (noisy, interconnected) set of business databases (employee performance, market trends, customer feedback, financial records), and a programming runtime. Day 0: the agent must read the databases and start operating.

**Each day.** The simulator advances time, deliverying new events: customer requests, market moves, employee actions, financial reports. The agent receives a daily summary, can query databases, write code to analyze them, and emit decisions: hire / fire, set prices, launch products, allocate budget.

**Coordination.** Decisions interact — pricing affects demand affects hiring affects budget. The agent must hold a coherent strategy across days, not just react locally.

**Horizon.** 500 days. The agent's actions on Day 5 keep affecting outcomes on Day 400.

**Scoring.** Programmatic. Final cash vs starting cash; subsidiary metrics (employee retention, customer satisfaction trajectories) for analysis. No judge LLM, no human rater.

## Why it matters

- **Cuts through the eval-saturation cycle.** Existing long-horizon benchmarks (Long-AgentBench, OS-World, GAIA) saturate within months of release. CEO-Bench's combination of long horizon × noisy info × coordination × programmatic scoring gives a hard, machine-graded target with real headroom.
- **The state of the art is bad.** Only Claude Opus 4.8 and GPT-5.5 (the day's frontier models) stay solvent over 500 days. Neither reliably turns a profit. The gap to competent human operators is large and quantifiable.
- **Right shape for the next eval phase.** As short-horizon benchmarks saturate, the field needs benchmarks where the bottleneck is *long-term coherence* rather than per-step capability. CEO-Bench is one of the first that fits.

## Gotchas & tricks

- **Compute cost.** 500 days × many decisions per day = a long agentic rollout. Running the benchmark on frontier models is expensive; expect papers to report on a small set of trials per model.
- **Stochasticity matters.** Different RNG seeds give different events; report mean ± variance over multiple seeds, not a single run.
- **Programming runtime is part of the API.** Agents that can write code to analyze databases win over agents that try to reason over raw text — this is a deliberate design choice favoring code-using policies.
- **Watch for memorization risk.** Since the simulator is deterministic given a seed, model providers could in principle train on dumped trajectories. Treat published numbers from training-data-overlap-suspect models carefully.

## Sources

- Paper: *CEO-Bench: Can Agents Play the Long Game?* — Chen, Liu, et al., Princeton, 2026 — [arXiv:2606.18543](https://arxiv.org/abs/2606.18543).
