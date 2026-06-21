# Structured State Ledger
*Depth — separate structured state + policy gate for policy-adherent tool-calling agents.*

**TL;DR:** Free-prompt agents lose track of accumulated task state and repeatedly violate policy constraints. **LedgerAgent** (Uddin, Saeidi, Blanco, Baral, ASU + U Arizona, arXiv 2606.20529) maintains a **separate structured ledger** of observed state (rendered back into the prompt each turn) and adds a **policy gate** that deterministically rejects state-dependent tool calls whose preconditions aren't met. Training-free, inference-time-only. Improves **pass^k** (k consistent successful trials) across four customer-service domains, with the biggest gains under stricter multi-trial consistency where free-prompt agents fall apart.

**Prereqs:** [README](README.md)
**Related:** [agentic-pipeline-optimization](agentic-pipeline-optimization.md) (different abstraction, complementary)

---

## What it is

Two coupled inference-time components:

1. **A structured ledger** — a typed key/value store of facts the agent has observed during the task. Examples: `customer_paid: true`, `refund_amount: 42.00`, `support_tier: gold`. Updated whenever a tool call yields a fact.
2. **A policy gate** — deterministic code that checks state-dependent preconditions before any environment-changing tool call. The gate consults the ledger, not the conversation transcript.

The agent's free-form chat memory is unchanged. The structured ledger is *additional* state, rendered into the prompt as a compact summary each turn.

## How it works

Per-turn flow:

1. Agent reads the conversation + the rendered ledger.
2. Agent proposes a tool call.
3. **Policy gate** intercepts:
   - If the call is read-only (no environment change) → execute, update ledger with the result.
   - If the call is state-changing → check ledger preconditions. If unmet, reject and return the violation to the agent. If met, execute and update the ledger.
4. Loop.

The ledger schema is task-specific (refund flow has different fields than account-management flow), but the gate logic is reusable.

## Why it matters

- **Bridges free-form LLMs and policy-strict production tasks.** Customer service, financial flows, regulated workflows all need deterministic policy adherence that free-prompt LLMs can't reliably provide.
- **No retraining required.** Drops onto any tool-calling LLM at inference time.
- **Best gains under strict consistency.** Pass^k metrics (the agent has to act correctly *every* time, not on average) are where free-prompt agents collapse and ledger-augmented agents win biggest.
- **Auditability for free.** The ledger is a structured log of what the agent observed and acted on — useful for compliance even before considering policy adherence.

## Gotchas & tricks

- **Ledger schema design is the load-bearing choice.** Too few fields and policy gates can't enforce constraints; too many and the rendered ledger eats prompt budget.
- **Rendering compactness matters.** Render only the ledger fields the agent might reference this turn (e.g. via a lightweight selector).
- **Gate rejection messages need to be useful.** A bare "policy violation" loops the agent on the same wrong action; include the unmet precondition so the agent knows what to fix.
- **Read-only vs write tool distinction must be reliable.** Misclassify a state-changing call as read-only and the gate gives no protection.
- **Does not replace training-time alignment.** Ledger + gate constrain *behavior*; they don't fix model misunderstanding of policy. Combine with policy-aware SFT for best results.

## Sources

- Paper: *LedgerAgent: Structured State for Policy-Adherent Tool-Calling Agents* — Md Nayem Uddin, Amir Saeidi, Eduardo Blanco, Chitta Baral, Arizona State University + University of Arizona, 2026, arXiv 2606.20529.
