# Executable Task Synthesis
*Depth — generate agent training tasks around a shared executable container state, with per-artifact repair instead of full re-rolls.*

**TL;DR:** A synthesized agent task is a bundle of four artifacts — *instruction*, *initial environment*, *reference solution*, *executable verifier*. Naive pipelines generate each in an independent pass and drift out of sync, producing unsolvable or wrongly-graded tasks. Executable task synthesis anchors all four in a single shared container state, then validates by *running* the reference solution against the verifier; per-artifact failures trigger targeted repair rather than regenerating the whole task. FACET reports consistent Terminal-Bench 2.1 gains from fine-tuning on synthesized trajectories.

**Prereqs:** [README.md](README.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [environment-reshaping.md](environment-reshaping.md)

---

## What it is

The training-data problem for agents mirrors the training-data problem for RLHF, but with a harder consistency requirement: for verifiable-reward RL to work, the *task* itself must be well-formed — a wrong verifier or an unsolvable initial state silently poisons the whole run. Executable task synthesis is the discipline of generating those four coupled artifacts so they refer to the same concrete world.

## How it works

The core loop, following FACET:

1. **Skill stitching.** Combine related agent skills (e.g., "grep this codebase", "edit config", "restart service") into an information-dense scenario. This is the *intent* the task will encode.
2. **Environment realization.** Build the container: filesystem contents, environment variables, running processes, network state. This becomes the shared grounding for everything else.
3. **Artifact derivation.** Read the *realized* container state to write the instruction, the reference solution (a sequence of commands / edits), and the executable verifier (typically a shell script or Python function).
4. **Execution validation.** Run the reference solution end-to-end. The verifier must return success. If it does not, tag which artifact is wrong (bad instruction? impossible solution? mis-scoped verifier?) and repair only that one.
5. **Trajectory collection.** Distill successful synthesis into (instruction, trajectory) pairs for downstream SFT or RL.

The invariant that makes this work: **the container is the single source of truth**. Instruction, solution, and verifier are all derived from it. Regeneration of one artifact does not have to re-derive the others.

## Why it matters

Verifiable-reward RL for agents is only as good as its tasks. Most current synthesis pipelines ship a substantial fraction of broken tasks — the failure modes (contradictory instruction, unreachable state, wrong verifier) are hard to spot without executing the reference solution against the verifier. Executable task synthesis makes broken-task rate observable and repairable, and the per-artifact repair loop cuts synthesis cost roughly proportional to the observed break rate.

The pattern generalizes: any executable-agent training corpus (browser tasks, code-repair tasks, database tasks) can be constructed the same way, swapping "container" for the appropriate execution substrate.

## Gotchas & tricks

- **Skip step 4 at your peril.** Without executing the reference solution against the verifier, a nontrivial fraction of tasks are silently broken. This is the biggest single quality lever in the whole pipeline.
- **Repair, don't re-roll.** Re-rolling on a per-artifact failure discards the valid parts and inflates cost. Isolate the broken artifact and regenerate only that.
- **Container hermeticity.** If reference solutions have side effects (e.g. write cache files), the initial state must be truly frozen — clone or snapshot before every run.
- **Verifier scope.** A verifier that checks too little accepts wrong solutions; one that checks too much rejects valid alternates. Prefer *executable behavioral* verifiers (does the service now respond correctly?) over *state-shape* verifiers (is this exact file present?).
- **Skill graph curation.** The initial "related skills" stitching drives task diversity. Random pairs produce artificial tasks; graph-based curation (co-occurrence in real repos, task-completion traces) produces tasks agents actually encounter.

## Sources

- Paper: *FACET: Preserving Source Intent and Executable State in Terminal Task Synthesis* — Shi, Wang, Su et al., 2026 — [arXiv:2608.18580](https://arxiv.org/abs/2608.18580).
- Related: *SWE-Rebench*, *TerminalBench* — benchmark ecosystems that motivated executable-artifact task pipelines.
- Related: *Rejection sampling for SFT data* — same "generate → verify → keep" spine, applied to responses instead of tasks.
