# Execute-Distill-Verify (EDV)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A separation-of-roles pipeline for agent experience learning. The agent that *executes* a task is not the one that *distills* lessons from it, and a third *verifier* must fail to refute each candidate lesson before it enters long-term memory. This breaks the self-confirmation trap where the same agent that ran a wrong trajectory certifies it as a success and re-uses it later.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md)
**Related:** [README.md](README.md) · [context-as-action.md](context-as-action.md) · [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

LLM agents that self-improve through experience write their own training signal: run a task, summarize what worked, store the summary for later retrieval. The failure mode is structural — a single agent has every incentive to file its own trajectory as "success" even when the underlying behavior was wrong but internally consistent. EDV decomposes the role into three agents with different incentives, each blind to the others' rationalizations.

## How it works

The pipeline runs three distinct agents per task:

| Role | What it does | Why it's separate |
| --- | --- | --- |
| **Executor** | Runs the task; produces a trajectory and a tentative outcome | Same incentive as today's agents |
| **Distiller** | Reads the trajectory and proposes candidate lessons (rules, heuristics, traps to avoid) | Cannot retroactively edit the trajectory; has no stake in the outcome |
| **Verifier** | For each candidate lesson, tries to refute it against independent evidence | Adversarial posture; only non-refuted lessons enter memory |

The verifier is the crucial piece. It is given the candidate lesson and asked to *find* a counter-example trajectory, citation, or constraint that would invalidate it. Lessons that survive enter the agent's long-term memory; those that don't are discarded. The verifier can be the same base model with a different system prompt, or a separately tuned model, or both in parallel.

This is structurally analogous to a Popperian falsification loop and to rejection sampling at the experience-learning layer.

## Why it matters

- Self-improving agents are a near-term reality, and contaminated memory is already a documented failure mode (the paper shows wrong-but-consistent trajectories accumulating into permanent "lessons").
- The fix is cheap — three model calls instead of one, no extra training.
- The separation-of-roles pattern generalizes: any time an LLM both produces and judges its own outputs, splitting the role across instances is a defense.

## Gotchas & tricks

- The verifier is only as good as its access to independent evidence. If all three agents share the same prior, none can falsify a coherent confabulation. Mitigation: give the verifier external tools (search, code execution).
- Cost scales 3× per task at experience-write time. Read time (retrieving stored lessons) is unaffected.
- Distinct from CoT monitoring ([../safety/cot-monitoring.md](../safety/cot-monitoring.md)) which inspects in-flight reasoning; EDV inspects post-hoc summaries before they become training data.

## Sources

- Paper: *Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning* — Zhu et al., 2026 — [arXiv:2606.24428](https://arxiv.org/abs/2606.24428).
