# Terminal Task Synthesis (FACET)
*Depth — synthesizing executable shell/CLI tasks for agent training with cross-artifact consistency guarantees.*

**TL;DR:** Training a terminal (bash/CLI) agent needs coherent quadruples: `(instruction, initialized environment, reference solution, executable verifier)`. If any pair is inconsistent, the task is silently unsolvable or wrongly graded — and multi-stage synthesis pipelines routinely drop the source material's goals, dependencies, and procedural constraints as they specialize each artifact. **FACET** (Fine-grained Agentic Construction of Executable Tasks) is a synthesis framework that (a) preserves information from the source and (b) enforces cross-artifact consistency so every generated task is both solvable and correctly verifiable.

**Prereqs:** [../data/_data-curation.md](../data/_data-curation.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

A task-synthesis pipeline for terminal agents that treats each task as a **4-tuple** and enforces two invariants across the tuple:

1. **Information preservation** — the goals, state transitions, dependencies, and procedural constraints in the source material survive into all four artifacts.
2. **Cross-artifact consistency** — the instruction is answerable, the reference solution satisfies the verifier when executed against the initialized environment, and the verifier accepts *only* solutions consistent with the instruction.

The output is a training set of executable tasks suitable for RLVR-style post-training on terminal agents.

## How it works

FACET breaks synthesis into small **agentic construction steps**, each of which:

- Ingests source material (docs, tutorials, one-off scripts, playbooks).
- Emits or refines one artifact of the 4-tuple.
- Threads a shared context capturing the source's dependencies and constraints.

Between steps, consistency checks run:

- The reference solution is executed inside the initialized environment; the verifier must accept it.
- Perturbed / adversarial candidate solutions must be rejected.
- The instruction is re-derived from the (env, solution, verifier) triple; drift from the original instruction is flagged.

Tasks that fail any check are dropped or repaired.

## Why it matters

Terminal agents (Codex-style copilots, shell-driving assistants) are one of the highest-leverage agent deployments. Their training data at scale is synthesized, and the failure mode "the RL task itself was wrong" is silent and pervasive — the policy either can't solve solvable tasks or gets credit for wrong ones. Frameworks that make task synthesis auditable directly raise the ceiling of agent-RL data quality.

## Gotchas & tricks

- Verifiers themselves must be expressive enough to reject all wrong solutions; a weak verifier will pass any of a family of tasks and produce spurious reward.
- Environment initialization is the easiest place for drift: an env that doesn't exactly match the state the instruction assumes will make the reference solution fail non-deterministically.
- Consistency checks add per-task cost; the paper reports acceptable overhead but scale-out demands parallel verifier execution.

## Sources

- Paper: *FACET: Preserving Source Intent and Executable State in Terminal Task Synthesis* — Shi, Wang, Su et al., 2026 — [arXiv:2608.18580](https://arxiv.org/abs/2608.18580)
