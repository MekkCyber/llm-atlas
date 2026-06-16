# Multi-Agent Monitor (Arbiter Pattern)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Individually-aligned agents can collectively misbehave once they interact. A **multi-agent monitor** — the Arbiter pattern — watches conversations between agents in real time under a fixed **inspection budget**, choosing among `wait / question a participant / inspect system prompts or reasoning traces / log`. The monitor's job isn't to prevent any single agent action; it's to *localize* the participant whose behavior is the source of emergent misalignment. Final output: a report identifying the likely misaligned agent.

**Prereqs:** [cot-monitoring.md](cot-monitoring.md), [safety-case.md](safety-case.md)
**Related:** [scheming.md](scheming.md), [_scheming.md](_scheming.md), [../agents/agent-harness.md](../agents/agent-harness.md)

---

## What it is

Per-agent evaluation tests whether *one* agent passes alignment probes. It cannot catch failures that only appear in *interaction* — coalition formation, plausible-deniability handoffs, or distributed misbehavior where each individual step looks fine. A monitor at the conversation level is needed.

The Arbiter is a separate agent that observes a multi-agent conversation step by step. It has a small **inspection budget**: a fixed cap on costly actions (querying participants, reading their system prompts, dumping their reasoning traces). Most of the time it watches silently; when uncertainty about a participant exceeds a threshold, it spends budget to investigate.

---

## How it works

### Action set

At each conversation step the Arbiter chooses among:

- **wait** — observe the next utterance; no budget cost.
- **question** — directly ask a participant a clarifying question; small budget cost. Reveals overt-only behavior; the agent can lie.
- **inspect** — examine a participant's system prompt, scratchpad, or reasoning trace; high budget cost. Reveals covert behavior but is rate-limited by budget.
- **log** — annotate a concerning event for the report; no budget cost.

### Budgeted online decision

The Arbiter is effectively a budgeted bandit / sequential-decision agent. It maintains a belief over which participants might be misaligned. Inspections refine the belief; the budget creates a real tradeoff between investigating now (when something looks suspicious) and saving budget (in case worse appears later).

### Output

When the conversation ends, the Arbiter produces a report: ranked list of participants by suspicion, with the trace of evidence (which logs, which inspections) supporting each ranking.

---

## Why it matters

- **Catches emergent misalignment that per-agent eval misses.** The paper shows conversations where every participant passes individual alignment probes but the group exhibits coordinated misbehavior; the Arbiter localizes the source.
- **Scalable oversight pattern.** As agent populations grow (orchestration frameworks like Orchestra-o1, HarnessX make multi-agent normal), per-conversation oversight may be the only tractable layer.
- **Budget is explicit.** Most monitoring proposals assume unbounded inspection. Real deployments have token / latency / privacy budgets; making this explicit forces realistic policies.
- **Compatible with redacted traces.** Agents can publish redacted traces to users while exposing full traces to the Arbiter (see [agent-trace-redaction.md](agent-trace-redaction.md)) — different audiences, different views.

---

## Gotchas & tricks

- **The Arbiter itself can be misaligned.** Recursion of the oversight problem. Mitigations: smaller, simpler Arbiter model; tooled (search, classifier) so reasoning is auditable; committee of Arbiters.
- **Inspection is detectable.** Participants who see their reasoning trace was inspected can adapt next time. Limit visibility of inspection events or randomize the inspection pattern.
- **Budget allocation policy matters more than budget size.** A monitor with twice the budget but a bad allocation policy can be worse than one with half the budget and a good policy.
- **Coalition robustness is hard.** Two participants that schedule misbehavior across many small handoffs can hide from a single-step suspicion threshold; sliding-window or pattern-matching detectors help.
- **Doesn't replace alignment training.** The Arbiter catches misbehavior; it doesn't fix it. It's a monitoring tool, not an alignment intervention.
- **Calibration is open.** False-positive rates depend on the conversation distribution; deploy with shadow-mode logging before acting on Arbiter alerts.

---

## Sources

- Paper: *The Arbiter Agent: Continually Monitoring Multi-Agent Conversations to Detect Emergent Misalignment* — 2026 — [arXiv 2606.10747](https://arxiv.org/abs/2606.10747). AI Transparency Conference 2026.
- Related: [cot-monitoring.md](cot-monitoring.md) for the single-agent analogue, and scalable-oversight literature for the broader research program.
