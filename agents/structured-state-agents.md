# Structured-State Agents
*Depth — separate the agent's belief state from its prompt context; use it as the substrate for both reasoning and policy enforcement.*

**TL;DR:** Standard tool-calling agents reconstruct task state (facts, identifiers, constraints) from the chat transcript on every turn, leaving them prone to grounding decisions in stale or missing facts and to syntactically valid tool calls that violate domain policy. **LedgerAgent** (2026) introduces an explicit **ledger** that records observed task state across turns, is rendered into the prompt as a separate section, and is consulted as a precondition check before any environment-changing tool call.

**Prereqs:** *(none — basic tool-calling agent literacy helps)*
**Related:** *(none in the graph yet)*

---

## What it is

A scaffolding pattern for tool-calling agents in policy-sensitive domains (customer service, ops automation, anything with side effects). The core move is to make **state explicit** instead of letting the LLM re-derive it from chat history every turn.

```
                    +---- ledger (explicit task state) ----+
                    | facts, IDs, constraints, conditions  |
                    +---------------+----------------------+
                                    |
                                    v
   user message  +-->  reasoning step  -->  proposed tool call
                                    |
                                    v
                       policy precondition check (vs ledger)
                                    |
                                    +--> blocked, ask LLM to revise
                                    +--> allowed, execute tool, update ledger
```

## How it works

LedgerAgent runs inference-time only (no training). Each turn:

1. **Read.** Render the ledger into the prompt alongside the chat history.
2. **Reason.** The LLM proposes a tool call as usual.
3. **Precondition check.** Domain policies are encoded as predicates over the ledger state. If a state-dependent constraint is violated, the call is blocked and the model is asked to revise.
4. **Execute & update.** On success, parse the tool response and append the new facts/identifiers/constraints to the ledger.

The ledger is the single source of truth for "what does the agent currently know?" — observations and tool returns can only modify it; the model reads from it the same way it reads from the chat history.

## Why it matters

- **pass^k** (probability of success across k independent trials) is the metric production agent teams actually need — single-shot accuracy is misleading. LedgerAgent's gains are largest under strict pass^k, meaning the failure mode it fixes is **variance**, not peak performance.
- Policy enforcement via preconditions is a clean substrate for compliance: domain rules are code over structured state, not prompt instructions the model may ignore.
- Works on closed-weight models (no finetuning needed), so it's deployable on top of frontier APIs.

## Gotchas & tricks

- The schema of the ledger is the engineering work. Too narrow and the LLM still grounds in chat history; too wide and the prompt fills with stale state.
- Precondition predicates have to cover the *actual* policies, not a paraphrase. Bugs here mean blocking valid calls (UX failure) or letting violations through (compliance failure).
- The ledger must be updated atomically with tool execution; partial updates leave the agent reasoning over an inconsistent state.
- Prompt budget. On long conversations the rendered ledger can grow large; summarization/eviction strategies matter for long-running agents.
- The pattern composes with — not replaces — function-calling schemas, retries, and reflection loops.

## Sources

- Paper: *LedgerAgent: Structured State for Policy-Adherent Tool-Calling Agents* — Uddin, Saeidi, Blanco, Baral, Arizona State / U. Arizona, 2026 — arXiv 2606.20529.
- Related: classical dialog-state-tracking literature (DSTC challenges).
- Related: τ-bench, AppWorld and other policy-adherent agent benchmarks.
