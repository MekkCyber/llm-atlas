# DarwinX
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Treats LLM-agent self-improvement as **evolutionary search over harnesses** — prompts, tools, skills, control flow — with the base model frozen. A **preserve-and-extend contract** only admits variants that add coverage without regressing existing tasks; an archive of alternative lineages enables recombination; and failure-, teacher-, and self-derived evidence share one edit interface. Fitness comes from each benchmark's own verifier.

**Prereqs:** [_post-training](../post-training/_post-training.md)
**Related:** [grpo](../post-training/grpo.md), [rlvr](../post-training/rlvr.md), [rl-prompt-curation](../post-training/rl-prompt-curation.md)

---

## What it is

An LLM agent's capability depends on model weights *and* on its **harness**: prompts, tool definitions, skill scripts, memory schemas, control flow. Single-lineage self-editing loops already exist — the model rewrites its own harness and hill-climbs — but they are path-dependent, and local gains on one task often regress others (the model overfits its harness to whichever eval it saw last).

DarwinX runs population-based search over harnesses with the model frozen. The contract of the search — what gets admitted, what gets kept, how variants mix — is where the design lives.

## How it works

**Population + fitness.** A population of harness variants runs against a benchmark suite. Each variant is scored by the benchmark's **own verifier** (test-pass rate, task-success rate). No gold solutions, no hand-picked winners.

**Preserve-and-extend contract.** A candidate variant is admitted only if it:

- **Extends** coverage on at least one benchmark task, AND
- **Preserves** performance on all previously solved tasks (no regressions).

This kills the "local win, global regression" failure of single-lineage editing.

**Archive of lineages.** Non-dominated variants are archived across generations, not just the current best. Later generations can **recombine** parts from different lineages (a prompt from one, a tool schema from another) — the equivalent of crossover in genetic algorithms.

**Tri-source edit interface.** Harness edits are proposed from three signals:

- **Failure evidence** — diffs from misclassified rollouts.
- **Teacher demonstrations** — external human or stronger-model traces.
- **Self-derived contrasts** — the agent comparing its own successful and failed runs on similar tasks.

All three go through one unified edit format, so the population operator doesn't care which source produced a candidate.

## Why it matters

Across four benchmarks that progressively separate the evolution signal from the test signal, **one evolutionary loop adds ~17 points on average**:

- **Terminal-Bench 2.1**: +7.7 → **83.2%** on a matched base; **84.7%** on a stronger base — the verified frontier.
- **TerminalWorld** held-out split: **68.3%**, ahead of every off-the-shelf agent.
- **WebArena-Infinity** real-task pass@1: **43.5% → 93.0%** audit-clean.
- A Terminal-Bench 2.1 harness **transfers unchanged** to SWE-bench Verified — evidence the evolved competence is general, not benchmark-specific.

Broader implication: if the harness carries substantial capability, training an agent doesn't require weight updates. Harness selection turns evaluation compute into **durable** capability. A frozen model need not be a fixed agent.

## Gotchas & tricks

- **The preserve-and-extend contract is what makes it work.** Loosen it (allow small regressions "to explore") and the population collapses into the same overfitting single-lineage editing produces.
- **Archive vs. active population.** Only the active population is evaluated each generation; the archive exists purely for recombination. Keeping it large is cheap but recombining too aggressively destabilizes.
- **Verifier quality is a hard floor.** DarwinX inherits its ceiling from the benchmark's verifier — noisy verifiers evolve noisy harnesses.
- **Compute profile is amortized eval, not training.** Cost scales with population × generation × per-run eval cost. No gradient. Runs on inference infrastructure.
- **Transferability isn't automatic.** The Terminal-Bench → SWE-bench transfer works because the evolved harness encodes general agent competence (planning, tool-use hygiene). Task-specific tool schemas won't transfer.

## Sources

- Paper: *DarwinX: Evolving Agent Harnesses Through Natural Selection* — Yifan Zhang, Yutong Dai, Juntao Tan, Luyu Yang, Rishi Mullur, Thai Hoang, Zhiyuan Hu, James Zhu, Phil Mui, Silvio Savarese, Ran Xu, Zeyuan Chen (Salesforce AI Research), 2026 — [arXiv:2608.07545](https://arxiv.org/abs/2608.07545).
