# Editable Skill Libraries for Agents
*Depth — persistent, structured skill packages that an agent can modify at deployment without any gradient step.*

**TL;DR:** Instead of fine-tuning an agent on new trajectory data, keep the model frozen and let a **skill library** evolve. Each skill is a multi-file package (retrieval metadata, executable plans, backup localization, failure-recovery rules, accessibility utilities, failure cases). At deployment, a **reflect-revise-reuse** cycle runs: the executor adjusts in real time, a critic diagnoses failures under information isolation, and the executor edits specific skill files through restricted tools. Introduced by EvoSkill-GUI (Chen et al. 2026) with +16.2% / +6.0% / +10.5% on MobileWorld / AndroidWorld / OSWorld.

**Prereqs:** [agent-harness](agent-harness.md), [_context-management](_context-management.md)
**Related:** [../evaluation/swe-bench.md](../evaluation/swe-bench.md)

---

## What it is

A **skill** = a directory of files describing one reusable capability (e.g. "log into the Settings app on Android", "install a Python package from source"). Typical contents:

- Retrieval metadata (name, tags, conditions for firing).
- Executable plan (step-by-step actions).
- Backup localization (fallback selectors when a widget moves).
- Failure-recovery rules (what to try on error X).
- Accessibility utilities (a11y hints for GUI actions).
- Failure cases (recorded past failures + fixes).

A **skill library** is a store of many such skills. The agent retrieves relevant skills into context, executes them, and — critically — is allowed to *edit* them after deployment via restricted tools.

---

## How it works

The **reflect-revise-reuse cycle**:

```
task ─► retrieve skills ─► executor runs plan ─► success? ─► done
                                                         │
                                                       failure
                                                         │
                                              critic diagnoses (isolated)
                                                         │
                                              executor edits specific skill files
                                                         │
                                              retry / cache for next time
```

Two design choices matter:

- **Information isolation between critic and executor.** The critic diagnoses failures without seeing the executor's later work; prevents the critic from unconsciously spoiling the fix.
- **Restricted edit tools.** The executor can only modify specific files (backup localization, failure-recovery rules) — the plan itself is edited under stricter conditions. Limits blast radius of a bad edit.

No fine-tuning, no rollouts, no reward model. Everything happens in the deployed inference loop.

---

## Why it matters

- **Alternative to trajectory-data SFT.** Fine-tuning GUI agents requires large, environment-specific trajectory datasets. Editable skill libraries add capability with none.
- **Composable with harness-layer optimization.** Sits above the harness — a good context manager retrieves the right skills; a good executor uses them.
- **Skills transfer across related tasks.** The paper shows evolved skills generalize to unseen but related tasks, not just the specific failure they were edited for.

---

## Gotchas & tricks

- **Skill store gets messy.** Without a curator, edits accumulate and skills contradict each other. Some periodic dedup/pruning is worth designing in.
- **Isolation is load-bearing.** Sharing state between critic and executor collapses the reflect-revise cycle into a self-consistency check.
- **Restricted tools = attack surface.** A jailbroken executor with permission to edit skill files is a persistence primitive — treat the tool interface as security-sensitive.
- **Cold start matters.** With an empty library the reflect-revise-reuse loop pays the cost of every failure. Seed the library with baseline skills for common tasks.

---

## Sources

- Paper: *Reflect, Revise, Reuse: Training-Free Skill Evolution for GUI Agents* — Chen et al., 2026 (Zhejiang U.) — [arXiv:2609.17653](https://arxiv.org/abs/2609.17653).
