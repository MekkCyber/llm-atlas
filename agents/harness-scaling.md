# Harness Scaling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The claim from StateM (2026): most long-horizon agent failures are runtime failures, not model failures — so scale the *harness* around an unchanged model instead of the weights. Concretely, replace a free-form ReAct loop with an execution frame that has (1) **durable state**, (2) **phase-local context**, (3) **checked transitions**, (4) **recoverable runbooks**, and (5) **versioned procedural practices**. On Terminal-Bench 2.1 this took GPT-5.5 xhigh from 83.1% to 92.1% and reached 95.3% raw accuracy overall, with one configuration reported as a $15 frontier-quality run.

**Prereqs:** [../agents/_agent-harness.md](_agent-harness.md)
**Related:** [agentic-transactions.md](agentic-transactions.md) · [black-box-rl-harness.md](black-box-rl-harness.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Harness scaling names the axis that agent research had been quietly using but not naming: *the same model gets qualitatively more capable when its runtime scaffolding gets better.* StateM formalizes that axis with a five-part contract for what a long-horizon agent runtime must offer.

The failure modes it targets, taken directly from the paper's motivation:
- The agent loses track of mutable state (files edited, env vars set, subprocesses spawned).
- It fails to reactivate lessons from an earlier execution (repeats mistakes across attempts).
- It skips known procedures the human operator would have followed.
- It stops prematurely because the loop's exit condition is under-specified.

Each of these is fixable at the runtime layer without touching weights.

## How it works

The five components of the StateM runtime:

1. **Durable state.** Named slots the agent must read from / write to. Not a scratchpad — a typed store the harness owns. Survives across turns and crashes.
2. **Phase-local context.** The prompt window in each phase is filtered to only the state and history relevant to that phase — no dumping the entire transcript back in.
3. **Checked transitions.** Moving from one phase to the next requires satisfying declared preconditions; illegal moves are rejected by the harness, not politely discouraged in the prompt.
4. **Recoverable runbooks.** A step that crashes doesn't restart the whole workflow — the runbook resumes from the last committed state, with the prior partial work intact.
5. **Versioned procedural practices.** Human-authored playbooks the agent must consult, versioned like code, inspectable by the user, and updated based on what recent runs failed on.

The model is unchanged. The **execution frame** carries the intelligence gains.

## Why it matters

- Reframes "long-horizon agent" as a systems problem, not a model problem. Sharpens the case that a benchmark score without a harness disclosure is a shared score for the harness and the model, not for the model alone.
- Puts a lower bound on cost: one reported StateM configuration hits frontier scores on Terminal-Bench 2.1 for **~$15**, because the durable-state + recoverable-runbook design avoids the cost of restarting failed runs from scratch.
- Explains why "small model + great harness" and "big model + weak harness" often come out even.

## Gotchas & tricks

- The state-machine authoring is where the engineering effort moves. Under-specify the phases and the model still drifts; over-specify and every new task needs bespoke harness work.
- Checked transitions can create livelock: the model repeatedly proposes moves the harness rejects. StateM's mitigation is to surface the reason for the rejection in the next prompt, not just the "no."
- Runbook versioning matters more than it looks. If practices change silently, later analysis of what caused a regression is impossible.
- "Harness scaling" is not "prompt engineering" — the point is that the *runtime* enforces the contract, not that the model is asked nicely.

## Sources

- Paper: *StateM: Reaching 95.3% Raw Accuracy, or a $15 Frontier Run, on Terminal-Bench 2.1 via Harness Scaling* — Ziheng Qin, Yaxin Lu, Zhangyang Atlas Wang, Kai Wang — arXiv:2608.15089 — 2026.
- Benchmark: Terminal-Bench 2.1 — the long-horizon terminal-task suite StateM reports on.
