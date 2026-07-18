# Coverage Map (multi-agent externalized progress)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An **externalized data structure** that tracks which sub-goals of a decomposable task have been completed vs. still open, shared across a multi-agent swarm. Introduced as a component of the SearchOS framework (2026) for open-domain information-seeking agents, but the pattern generalizes to any long-horizon task where multiple agents work in parallel and progress needs to be visible outside any single agent's context.

**Prereqs:** *(none — the `agents/` folder currently has only a README; this is the first depth file)*
**Related:** *(none in KG yet)*

---

## What it is

Multi-agent systems typically let each agent hold its own private context. When one agent fails on a sub-goal, its failure isn't visible to the swarm; when several agents overlap on the same sub-goal, they each redo the work. A **Coverage Map** externalizes progress state into a shared object with a *specific* structure:

- The task is decomposed into **cells** — atomic sub-goals whose completion is checkable (a specific attribute of a specific entity, a specific evidence field, a specific tool-call output).
- Each cell has a **status**: *unresolved*, *in-flight* (some agent is working on it), or *resolved* (with a citation to the evidence that resolved it).
- Every agent reads and writes the coverage map through a scheduler; agents don't privately mutate progress state.

The map is the shared "what's done" so that the scheduler can dispatch idle agents against *unresolved* cells rather than re-dispatching them against cells another agent is already working on.

## How it works

In SearchOS, the coverage map is one of four externalized structures (Frontier Task, Evidence Graph, Coverage Map, Failure Memory). They interoperate:

1. **Task decomposition.** Open-domain information seeking is reformulated as *relational schema completion with grounded citations* — the task becomes a set of tables whose cells are (entity, attribute) pairs. Each cell is one coverage-map slot.
2. **Scheduler dispatch.** A pipeline-parallel scheduler picks unresolved cells based on urgency and dependency (some cells depend on others being resolved first) and hands them to idle agents.
3. **In-flight marking.** Before an agent starts a cell, the scheduler marks it *in-flight* — a lightweight lock to prevent duplicate work.
4. **Resolution or failure.** When the agent returns evidence, the scheduler resolves the cell with a citation and updates the Evidence Graph. If the agent fails or times out, the scheduler consults Failure Memory to avoid dispatching the same failing search pattern again, then re-opens the cell.
5. **Freed slots refill.** As cells resolve, the scheduler refills freed agent slots with the next-highest-priority unresolved cells — utilization stays high even as easy cells finish quickly.

## Why it matters

- **Turns fragile implicit progress into inspectable state.** Any human overseer can look at the coverage map and see exactly what's done vs. left — a huge win for auditability and debugging.
- **Eliminates duplicated work.** Multi-agent swarms without shared state routinely redo cells; the map makes overlap impossible by construction.
- **Enables pipeline-parallel scheduling.** With a queue of unresolved cells and a pool of agents, the scheduler can keep utilization high without agents having to coordinate directly.
- **Generalizes beyond information seeking.** Any long-horizon task with a checkable decomposition — code refactors, research synthesis, database migrations — can use the pattern. The specific data structure (relational schema completion) is the SearchOS instance; the general pattern is externalized checkable sub-goals.
- **Wins on WideSearch and GISA.** SearchOS-V1 leads all metrics vs. evaluated single- and multi-agent baselines on those benchmarks.

## Gotchas & tricks

- **Decomposition is the hard part.** The coverage map is only as useful as the task decomposition it tracks. For open-domain tasks, forming a *good* schema of what needs to be resolved is itself an agent skill — SearchOS treats schema induction as the first step.
- **Cell granularity matters.** Too-fine cells cause scheduler thrash (every dispatched agent finishes in one action); too-coarse cells cause under-utilization (idle agents while one big cell blocks). Empirically-driven cell design.
- **In-flight locks need timeouts.** An agent that dies mid-cell holds a lock forever; timeout + re-open is required.
- **Coverage is not correctness.** A resolved cell has *some* evidence attached, not necessarily *right* evidence. Downstream verification is still needed. The map avoids missing work, not wrong work.
- **Externalized state must be serializable.** For the map to survive agent restarts (or handoffs to different runtimes), the schema and its evidence citations need a stable serialization.
- **Failure Memory is a sibling structure.** Coverage tracks *what's still open*; Failure Memory tracks *what has already failed and how*. Both are needed — one avoids duplicate work, the other avoids repeating known-failing search patterns.

## Sources

- Paper: *SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration* — Gao et al., Renmin U. / Ant Group, 2026 — introduces the coverage map as part of Search-Oriented Context Management (SOCM).
