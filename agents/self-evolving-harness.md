# Self-Evolving Agent Harness
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A framework for improving an agent **with its weights frozen**: evolve the surrounding software — prompts, memory, workflows, callable tools, repository-level control logic, plugin composition — based on validation feedback. Formalizes *what* may change, *who* proposes changes, and *how* admissions are controlled with continual PAC-learning guarantees on cumulative regressions and acceptance probability.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md), [../safety/scheming.md](../safety/scheming.md)

---

## What it is

Most agent improvement in deployment can't retrain weights (closed API, local-only inference, cost). The remaining lever is the harness: everything the model reads and everything it writes to and through. Self-evolving harnesses treat this as the primary unit of update:

- Harness state (prompts, memory contents, learned examples).
- Harness code (tool implementations, workflows, retry logic).
- Harness composition (which plugins are loaded, in what order).

Changes are proposed by an outer-loop process (another LLM, evolutionary search, user feedback) and admitted only after passing a validation test. Admitted revisions persist across runs — the harness genuinely changes.

---

## How it works

Three dimensions, factored:

### 1. Change target (what evolves)

- **Interpreted state**: prompts, memory entries, context exemplars — editable text.
- **Tools and code**: callable functions and their implementations.
- **Repository-level control**: workflow DAGs, retry policies, routing rules.
- **Plugin composition**: which modules are loaded.

### 2. Proposal mechanism (who proposes)

- **LLM-as-editor** — outer-loop model reads failure traces and emits revisions.
- **Evolutionary search** — mutate harness fields, select by fitness.
- **User feedback** — explicit prompt / tool edits from the operator.

### 3. Admission rule (how a change is accepted)

Candidates are evaluated on a held-out validation set drawn from recent task distribution. The paper derives, under continual PAC-learning assumptions:

- A **lower bound on validation set size** needed for expected generalization at a chosen admission threshold.
- A **cumulative regression bound** — across many sequential admissions, the probability of accumulated regression stays bounded.
- A **high-probability admission** criterion that acts as a stopping rule for the outer loop.

Rejected candidates are discarded; accepted ones replace the current harness field atomically.

---

## Why it matters

- **Matches the deployment reality.** Frontier agents today ship as frozen-weights services; the field already updates them via prompt / tool / workflow edits ad hoc. Self-evolving harnesses are what you get when you make that loop principled.
- **Reversible and auditable.** Weight-based improvement is opaque and hard to roll back; harness revisions are text diffs with validation evidence. Operationally, this is a safer update path.
- **Composes with weight-based training.** Not an alternative to RL/SFT — a different update layer. Terminal-RSI stacks weight updates on top of harness evolution (see [recursive-self-improvement](./recursive-self-improvement.md) once added).
- **Opens a research surface.** Many obvious follow-ups: safety of self-rewrite (can an evolved harness route around operator policies?), transfer of harness improvements across agents, interaction with scheming behavior.

---

## Gotchas & tricks

- **Validation distribution drift.** PAC bounds assume the validation set reflects the task distribution. In production the task mix shifts; harness revisions tuned on stale validation can regress on new tasks. Rolling / weighted validation sets mitigate.
- **Spurious admissions compound.** Each accepted change is a chance to overfit. Keep an orthogonal held-out "golden" set that is never used for admission.
- **Tool-code changes are riskier than prompt changes.** A new tool implementation can introduce side effects outside the validation suite's coverage. Lock down code changes behind stricter admission thresholds.
- **Not a free pass on safety.** A harness that can rewrite its own control logic has new failure modes — scheming via self-edit is a legitimate concern. Independent safeguards (immutable audit logs, operator-only approval of code changes) are needed for deployed systems.

---

## Sources

- Paper: *Self-Evolving Agent Harness* — Wei et al., Zhongguancun Academy / USTC / Zhongguancun AI Institute, 2026 — arXiv [2610.18813](https://arxiv.org/abs/2610.18813).
- Related: *Terminal-RSI* — Jiang et al., 2026 — arXiv [2610.22192](https://arxiv.org/abs/2610.22192) — pairs harness self-rewrite with on-policy weight RL.
