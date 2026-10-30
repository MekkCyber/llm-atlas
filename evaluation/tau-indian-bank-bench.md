# TauIndianBankBench
*Depth — a procedure-aware tool-use benchmark for conversational agents in Indian retail banking.*

**TL;DR:** **TauIndianBankBench** (Adhikary & Devadiga, NPCI, 2026) is a benchmark of **1,000 simulated Indian retail-banking sessions** that grades a tool-using LLM agent on both the final database state **and** the procedural steps taken. For every tested model, **6.5–12.0% of sessions** that count as outcome successes under TauBench-style scoring are **procedural failures** — the assistant skipped a required step or made an unrequested change. The benchmark is one of the first serious agent evals produced by a national payments-infrastructure body (NPCI), targeting regulated enterprise deployment.

**Prereqs:** *(none)*
**Related:** [../agents/procedure-aware-eval.md](../agents/procedure-aware-eval.md), [_agent-benchmarks.md](_agent-benchmarks.md)

---

## What it is

A tool-use benchmark where the agent plays a bank's digital assistant:

- 1,000 simulated sessions.
- Realistic Indian-retail-banking tasks (transfers, limit changes, dispute initiation, KYC updates).
- Agents interact with a mock bank API with full database and audit-log state.
- A scripted user supplies account context and in-session responses.

What makes it different from TauBench: **procedure-aware grading**.

### Scoring axes

| Axis | What it checks |
| --- | --- |
| **Outcome** | Final DB state matches the ground truth |
| **Procedural compliance** | Every required step (identity verification, limit confirmation, consent capture) happened |
| **Non-intervention** | No *unrequested* database changes were made |
| **Audit trail** | The required audit entries exist |

An agent that reaches the correct final state while skipping a verification step scores a procedural failure, even if the outcome is "right."

---

## How it works

### Session structure

```
scripted user turn ──► agent turn
                       │
                       ├─ tool call (bank API)
                       └─ response
```

The bank API returns both normal responses and audit-log entries. The grader has access to both the final DB state and the complete audit log.

### Procedure graph

Each task class has a **required-step graph**: a partial order of steps that must be observed before any state-modifying tool call. Example:

```
account_transfer:
  verify_identity → check_balance → confirm_amount → execute_transfer
                                 ↘   check_daily_limit  ↗
```

A trace that reaches `execute_transfer` without `confirm_amount` fails procedural compliance.

### Non-intervention

The grader logs every state-modifying API call made during the session and checks each against the user's expressed intent. Unexpressed changes (e.g., auto-enabling a feature while adjusting a limit) are flagged.

---

## Why it matters

- **Regulated deployment needs procedure, not just outcome.** Banking, healthcare, legal: compliance means the steps were taken, not just that the final state is right.
- **The 6.5–12% silent-failure band is invisible to TauBench-style scoring.** Every model tested has non-trivial silent failures. For regulated use, that's catastrophic.
- **Benchmark produced by a payments infrastructure body (NPCI).** Signals that national regulators are taking agent evaluation seriously.
- **Transferable schema.** The procedure-graph + non-intervention + audit-trail pattern transfers to any regulated enterprise domain.

---

## Gotchas & tricks

- **Procedure graphs are domain-specific.** You cannot copy the schema to healthcare without re-authoring the graphs. The methodology transfers; the artifacts don't.
- **Non-intervention grading is binary and strict.** A legitimately-user-benefitting but technically unrequested change still counts as a failure. This is deliberately conservative.
- **Agent CoT must be ignored during grading.** Only tool calls and final state count — a model that *says* it will verify identity but never calls `verify_identity` fails. This is intentional: CoT is not a compliance artifact.
- **Scripted user is NPCI-specific.** User responses reflect Indian retail banking conventions; some may surprise agents trained on US-English data.
- **Not a stress test for conversational quality.** Collaboration-style issues (over-asking, poor explanations) are not scored here. Pair with [cli-agent-bench.md](cli-agent-bench.md) or TauBench for that axis.

---

## Sources

- Paper: *TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents* — Adhikary & Devadiga, National Payments Corporation of India (NPCI), 2026, [arXiv 2610.30161](https://arxiv.org/abs/2610.30161).
- Related: TauBench (2024) — the outcome-only tool-use benchmark predecessor.
- Related: [../agents/procedure-aware-eval.md](../agents/procedure-aware-eval.md) — the broader concept.
- Related: [_agent-benchmarks.md](_agent-benchmarks.md) — benchmark-landscape taxonomy.
