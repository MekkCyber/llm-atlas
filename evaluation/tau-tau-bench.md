# τ^τ-Bench (Hyper-Tau-Bench)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** τ^τ-bench turns **agent construction** into the benchmark task. A developer agent inherits the records, live API, codebase, client requirements, and budget of a real customer-service engagement, and must deliver an agent that passes when deployed against held-out simulated users. Across 53 tasks in 4 domains, the strongest configuration (Claude Opus 5 under Claude Code) reaches 23.9% vs an 82.2% expert-authored ceiling.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [humaneval.md](./humaneval.md), [livecodebench.md](./livecodebench.md)

---

## What it is

τ-bench (2024) evaluates a **single agent** on tool-use tasks. τ^τ-bench moves up one meta-level: the evaluee is a *coding agent* whose job is to **build** the customer-service agent that τ-bench would then evaluate. This mirrors real deployment work — most production LLM agents are shipped by an engineer (increasingly, another agent) rather than plugged in directly.

The starting materials for each task are the ones a real engagement provides:

- The **business records** the deployed agent will need to reason over.
- A **client** simulator that holds requirements and must be interrogated.
- A **production API** the deployed agent must run through.
- An **inherited codebase** with existing (often ugly) helpers and structure.
- **Budget limits** on serving cost and models.

The developer agent's deliverable is a working customer-service agent. Scoring runs *that* agent against a held-out simulated user population and measures pass rate.

## How it works

- **53 tasks** across 4 domains, each with a distinct business schema and API.
- Deliverable is *code* (an agent implementation), not a chat transcript.
- Evaluation batches deployed agents against a simulated-user pool; pass criterion is task-specific but always end-to-end (the user's request must be resolved).
- **Ceiling.** Expert-authored reference implementations score 82.2% — establishing that the benchmark is winnable and calibrating the ceiling.
- **State of the art.** Claude Opus 5 under Claude Code: 23.9%. Weaker configurations trail sharply.

Diagnosed failure modes of frontier coding agents:

- **Shallow record queries.** Look at a handful of examples, generalize to the domain, miss corner cases.
- **Silent client interaction.** Ask the client almost nothing; ship the first plausible interpretation of the requirements.
- **Premature architectural commitment.** Adopt the first design that runs; do not experiment with alternative agent architectures or spend budget on parallel prototypes.

## Why it matters

- **Puts agent construction on the scoreboard.** Previously "the coding agent built a working agent" was anecdotal / vibes. τ^τ-bench turns it into a measurable, comparable capability.
- **Aligned with real economics.** Building agents is where most of the near-term commercial value of AI coding tools sits. A benchmark that matches the shape of that work steers model development toward it.
- **Exposes coding-agent weaknesses beyond code.** The three diagnosed failure modes (shallow queries, poor client comms, premature commitment) are *human*-agent process failures, not code-writing failures — so improving code eval scores alone will not close the gap.

## Gotchas & tricks

- **Ceiling is calibration, not floor.** 82.2% expert score means the task set is not saturated by humans either — a 100% target is unrealistic.
- **Configuration matters as much as model.** "Claude Opus 5 under Claude Code" is not just a model choice; the harness's memory, tool set, and planning discipline drive a large fraction of the score. Report the full stack.
- **Long horizons, expensive runs.** Each task is a full engagement; expect hours of wall-clock and non-trivial serving spend for a full 53-task pass.
- **Simulated users are not real users.** Failures on simulated populations are lower-bound estimates of real-user failure.

## Sources

- Paper: *τ^τ-Bench: An Environment for End-To-End, Realistic Agent Construction* — Quan Shi, Keshav Dhandhania, Karthik Narasimhan, Victor Barres — arXiv:2609.04611 (2026).
- Paper: *τ-bench* — Yao et al., 2024 — the single-agent tool-use predecessor.
