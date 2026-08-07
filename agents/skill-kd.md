# SKILL-KD: Contrastive Skill Distillation

*Depth — a prompt-time distillation method that turns teacher/student trajectory diffs into version-controlled skill patches for frozen students.*

**TL;DR:** Skill libraries for LLM agents (Voyager-style) accumulate silently and drift. SKILL-KD (Shi et al., 2026) treats a skill as an explicit *patch*: given a student failure and a teacher success on the same task, extract their actionable discrepancy as a textual rule, re-run the student to test it, and iteratively refine when the student still fails. **Drift-Aware Skill Consolidation** decides per-patch whether to add, delete/modify, or skip a rule against a trace-linked edit history. Works with frozen students — no parameter updates.

**Prereqs:** [../post-training/README.md](../post-training/README.md)
**Related:** [agent-harness.md](agent-harness.md) · [../post-training/on-policy-distillation.md](../post-training/on-policy-distillation.md)

---

## What it is

Prior skill libraries treat skills as memory entries — successful demonstrations summarized and cached for retrieval. This has a mismatch for weaker students: when a student fails because it lacks *task knowledge* or *operational strategy*, the failed trajectory alone can't reveal what's missing, and the teacher's trajectory is too implicit to internalize as reusable guidance.

SKILL-KD makes the skill the *diff* between them.

## How it works

1. **Failure pair.** Collect a task on which the student fails and the teacher succeeds. Store both trajectories.
2. **Skill patch extraction.** An extractor (typically the teacher itself) writes a short textual rule capturing what the teacher did that the student did not — an if/then guideline or an explicit sub-step order.
3. **Verify by re-run.** Insert the patch into the student's prompt / skill library and re-run the same task. If the student now succeeds, accept; if not, refine the patch and repeat (bounded iteration cap).
4. **Drift-Aware Skill Consolidation.** Each accepted patch is compared to the existing skill library. Decide per-patch: **add** (new rule), **modify** (edit an existing rule), **delete** (retract a rule the new patch contradicts), or **skip** (redundant). The trace-linked edit history is what enables this decision — every patch remembers which failure it fixed.

The library is a version-controlled artifact, not an accretive dump.

## Why it matters

- **Turns skill libraries into version-controlled patches.** Consolidation prevents the classic "prompt library rot" where old rules contradict new ones and the agent behavior drifts unpredictably.
- **Frozen-student compatible.** No weight updates needed — this is deployment-friendly for closed-source APIs.
- **Verifiable per-patch.** Every patch has a specific task and re-run outcome tied to it; regressions are locatable.

## Gotchas & tricks

- **Patch extraction quality bounds everything.** A weak extractor produces patches that memorize surface features of the failure task and don't generalize. The teacher (or a stronger auditor) should extract.
- **Consolidation heuristics are hyperparameters.** "When is a new patch redundant with an existing one?" is a similarity-threshold judgment; wrong threshold either bloats the library or clobbers useful rules.
- **Frozen-student ceiling.** Patches encode strategy but cannot teach missing base capabilities — if the student lacks arithmetic, no patch will fix that.
- **Cross-task transfer is untested at scale.** The paper shows within-benchmark gains; broader transfer of the skill library across benchmarks is future work.

## Sources

- Paper: *SKILL-KD: Contrastive Skill Distillation for LLM Agents* — Shi, Dou, Zhu, Tao, Jin, Kang, Zhou, Weng, 2026 — [arXiv 2607.28048](https://arxiv.org/abs/2607.28048).
