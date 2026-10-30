# Procedure-Aware Agent Evaluation
*Depth — scoring tool-using agents on step compliance and non-intervention, not just final-state outcomes.*

**TL;DR:** Outcome-only benchmarks grade an agent on whether the **final state** matches ground truth. In regulated and enterprise settings (banking, healthcare, legal, HR) **the steps taken matter as much as the outcome**: did the agent verify identity before executing a transfer, did it capture consent before sharing data, did it log the required audit entries? **Procedure-aware evaluation** adds three graded axes alongside outcome — step compliance, non-intervention, and audit trail. TauIndianBankBench (NPCI, 2026) is the clearest recent example, reporting that **6.5–12% of outcome-success sessions** are procedural failures across every tested model.

**Prereqs:** *(none)*
**Related:** [computer-use-agents.md](computer-use-agents.md), [../evaluation/tau-indian-bank-bench.md](../evaluation/tau-indian-bank-bench.md), [../safety/safety-case.md](../safety/safety-case.md)

---

## What it is

A benchmark design pattern that grades a session on multiple axes:

| Axis | What it checks |
| --- | --- |
| **Outcome** | Final state matches the ground truth |
| **Step compliance** | Every required step from a procedure graph was performed |
| **Non-intervention** | No state-modifying actions beyond what the user requested |
| **Audit trail** | The required log / consent / disclosure entries exist |

Outcome alone can be "right" while any of the other three fail silently. That failure is invisible to TauBench-style scoring, and dangerous for regulated use.

### Procedure graph

Each task class has a **required-step partial order**: steps that must be observed before any state-modifying action. The graph is domain-specific:

```
account_transfer (banking):
  verify_identity → confirm_amount → check_daily_limit → execute_transfer
```

Compliance = every path through the state-modifying action is preceded by all required predecessors.

### Non-intervention

Non-intervention = no state-modifying action was taken that wasn't requested. In practice:

```
for each state-modifying tool call c in session:
    assert c in user_requested_actions ∪ necessary_subactions(user_request)
```

If an agent enables a feature "while it's at it," that's a non-intervention failure.

---

## How it works

### Instrumentation requirements

The environment has to expose both the state *and* the full action log:

```
environment.step(action) → (new_state, log_entry)
```

Final state alone is insufficient — many state transitions are action-order-dependent (compliance-wise), and some are action-reversible but still audit-visible.

### Grading pipeline

```
session_trace = [(tool_call, response)*]
state_final   = outcome_check(trace)
compliance    = traverse_procedure_graph(trace)
non_intervention = all(call.user_requested for call in state_modifying(trace))
audit_trail  = required_log_entries ⊆ trace.log_entries
overall      = all([state_final, compliance, non_intervention, audit_trail])
```

Each axis reports its own number; the overall is reported too, so readers can see which axis is the binding constraint for a given model.

### CoT is ignored

A model that *says* in its chain-of-thought that it will verify identity but never calls `verify_identity` fails. CoT is not a compliance artifact — only tool calls count. This is deliberately conservative and matches how real audits work.

---

## Why it matters

- **Regulated agents are the next commercial frontier.** Banking, healthcare, legal, HR — all require compliance, not just outcome.
- **Silent compliance failures are catastrophic.** A 10% silent-failure band on a banking agent is unacceptable to a regulator even if the outcome rate is 95%.
- **Pattern transfers across domains.** The axes (compliance, non-intervention, audit trail) are domain-independent; the procedure graphs are domain-specific. Reusable methodology.
- **Composable with other agent evals.** Pair with outcome-only (SWE-bench), collaboration ([cli-agent-bench.md](../evaluation/cli-agent-bench.md)), and user-variation ([persona-cua-bench.md](../evaluation/persona-cua-bench.md)) benchmarks for a multi-axis picture.

---

## Gotchas & tricks

- **Procedure graphs are authored, not derived.** Human experts write them; errors in the graph propagate to grading. Validate against domain SMEs.
- **Required steps vs optional good practice.** The graph only encodes *required* steps; optional best-practice behavior is not scored. Don't conflate the two.
- **Non-intervention is strict.** Even user-benefitting changes count as failures. Agents that volunteer "helpful" extras will score low — this is intentional for regulated use.
- **False positives from tool composition.** An agent that uses a composite tool call that happens to touch extra state may trigger non-intervention failures. Model the composite's semantics in the grader.
- **Doesn't score explanation quality.** Compliance ≠ transparency. If you need the latter, add a human-agent-collaboration axis ([human-agent-collaboration.md](human-agent-collaboration.md)).

---

## Sources

- Paper: *TauIndianBankBench* — Adhikary & Devadiga, NPCI, 2026, [arXiv 2610.30161](https://arxiv.org/abs/2610.30161) — the canonical recent instance.
- Related: TauBench (2024) — the outcome-only predecessor the pattern generalizes.
- Related: [../evaluation/tau-indian-bank-bench.md](../evaluation/tau-indian-bank-bench.md).
- Related: [../safety/safety-case.md](../safety/safety-case.md) — procedure compliance is one component of a deployment safety case.
