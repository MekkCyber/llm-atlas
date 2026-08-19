# Agentic Transactions
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long-horizon agent workflows fail like partially-applied database transactions: the workflow crashes halfway, the world is now inconsistent (email sent but calendar not updated, PR opened but tests not queued), and cleanup is manual. *Agentic Transaction* (Tsinghua, 2026) reinterprets database **ACID** for LLM agent systems and wraps agent execution in a transaction manager that gives Semantic Atomicity, Consistency, Isolation, and Durability over multi-step tool calls, reporting a 10.6% improvement over comparable systems.

**Prereqs:** [_agent-harness.md](_agent-harness.md)
**Related:** [harness-scaling.md](harness-scaling.md) · [../systems/README.md](../systems/README.md)

---

## What it is

Databases spent 40 years learning that "partial success" is worse than "no success" — either the whole transaction commits, or it rolls back to a consistent state. Agents in production violate that constantly. Agentic Transaction lifts the classical ACID contract into the semantic domain of tool calls, where "rollback" isn't just deleting a row but compensating an already-observable side effect (unsend the email, cancel the calendar invite, revert the commit).

The unit is the **agentic transaction**: a scoped workflow of tool calls whose external effects are managed as a single logical operation.

## How it works

Four adapted primitives:

1. **Semantic atomicity.** If the workflow can't finish (crash, veto, budget cap), its already-executed tool calls are compensated by matched **inverse actions** the harness registers alongside each tool. Not deletion — *tool-level* compensation (e.g., `send_email` pairs with a `retract_email` compensator; irreversible tools are refused until the workflow is proven able to complete).
2. **Semantic consistency.** The workflow declares invariants over external systems (e.g., "the calendar and the email must agree on the meeting time"). The manager checks invariants at commit and reports violations.
3. **Semantic isolation.** Parallel agent workflows are scheduled so their observable effects don't interleave in ways that would leave any single workflow with an inconsistent view. Classic concurrency control, translated to tool calls.
4. **Semantic durability.** Once the manager reports commit, side effects survive process crashes — the manager persists a commit log that a restarted agent can replay or reconcile against.

The user-facing surface is a decorator or context manager wrapping a block of tool calls; the manager handles scheduling, logging, compensation, and commit.

## Why it matters

- Agents are being deployed against real systems where partial failure has real cost. ACID-for-agents is the missing systems abstraction between "the agent tried" and "the world is consistent."
- Reframes agent reliability: instead of "make the model retry harder," build the runtime such that a mid-workflow failure is *recoverable by construction*.
- Makes irreversible tools first-class citizens. If a tool has no compensator, the manager refuses to touch it until the workflow's success is proven — a much stronger safety property than "please double-check."

## Gotchas & tricks

- **Compensation is hard to write.** Some tools genuinely have no inverse (send a physical package, publish a tweet, pay an invoice). The manager can only reject or quarantine those; it can't fake atomicity.
- **Isolation levels matter.** Full serializability across agents is expensive and often overkill. Weaker isolation (snapshot, read-committed) may be enough, but you have to *choose* — silent lock is the default failure mode.
- **Invariants are user-authored.** If nobody declares the "calendar and email must agree" invariant, the manager can't check it. Ships best when combined with a workflow-authoring UI that surfaces missing invariants.
- **Do not confuse with retries.** Retrying a failed step re-runs it; a rollback undoes the effects of a step that already ran. Agentic transactions need both, in the right order.

## Sources

- Paper: *Agentic Transaction: Towards ACID-Compliant Agent Systems* — Zhaoyan Sun, Xiaoxiao Wang, Guoliang Li — arXiv:2608.13900 — 2026 (Tsinghua University).
- Textbook reference: Gray & Reuter, *Transaction Processing: Concepts and Techniques* — the ACID primitives being lifted.
