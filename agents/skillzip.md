# SkillZip
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An **evaluation-free** method for compressing the accumulated "skills" of a self-evolving agent. Formalizes compression as a **typed minimum-description-length** objective over a skill *contract* + *residual*, with a hard coverage constraint on every trigger, workflow edge, tool requirement, obligation, and output field. Ships in two modes: **one-shot** (single structured extraction + deterministic optimization) and **continual Zip-on-Write** (folds each self-evolution patch in incrementally without replaying tasks). Introduced in *SkillZip* (2026).

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

Self-evolving agents accumulate reusable "skills" — structured procedures with triggers, workflow steps, tool contracts, and rare-but-essential exceptions. Over time the same requirement is restated across branches, examples, and warnings; common action sequences are copied rather than referenced. The skill inflates, injection cost balloons, maintenance becomes fragile.

Generic prompt compression is unsafe here because a skill is not flat prose:

- **Name + description** gate when the skill applies (drop them and the skill activates on the wrong tasks).
- **Workflow** controls execution (compressing it can change behavior).
- **Tool + output contracts** constrain validity.
- **Rare exceptions** may be essential even when no sampled task activates them (so evaluation-driven compression can silently delete them).

**Evaluation-guided compression** solves some of this by rollouts on a test set — but that introduces cost, dependency on the eval set, and no guarantee for out-of-distribution triggers.

SkillZip is **evaluation-free**: it compresses purely by finding the skill's shortest faithful *structural* explanation.

## How it works

**Formulation.** A skill is decomposed into a **typed contract** (declarative structure — triggers, workflow edges, tool requirements, obligations, output fields) and a **residual** (any freeform text that isn't captured by the contract). The objective is a **minimum-description-length** loss over (contract, residual), subject to a **hard coverage constraint**: every trigger, workflow edge, tool requirement, obligation, and output field in the original must be represented in the compressed version.

**Intuition — "explain once, reference many":**
- State a repeated rule **once at the scope where it applies**.
- Factor a repeated action sequence into a **shared procedure**.
- Keep only the **differences as explicit exceptions**.

**Sharing thresholds** are simple: promote a rule when it applies to ≥ N sibling nodes; extract a procedure when it appears ≥ M times.

**Two modes:**
- **One-shot.** A single structured extraction call turns the skill text into the typed contract; deterministic optimization then finds the minimum-length representation. No task rollouts required.
- **Continual Zip-on-Write.** Each new self-evolution patch is folded into the existing compressed skill by re-running the local optimization over the affected sub-tree. No replay, no reparse of the full history.

## Why it matters

- **Removes the rollout tax.** Prior compression required test rollouts on every candidate; SkillZip is deterministic once the contract is extracted.
- **Preserves rare-but-essential rules by construction.** The hard coverage constraint means unique exceptions cannot be silently dropped.
- **Continual mode fits real deployments.** Agents that keep learning need incremental compression, not a periodic re-run over the full history.
- **Direct cost impact.** Skill text dominates the context of any long-running skill-based agent; compressing it 5–10× drops per-call token cost proportionally.

## Gotchas & tricks

- **Contract quality is the ceiling.** A poor initial contract extraction (e.g. missing an output field) breaks the coverage guarantee. Validate the extraction on a small held-out set before trusting the compression.
- **Freeform residual can hide reasoning.** If too much ends up in the residual, the coverage guarantee applies only to the typed parts. Aggressively push structure into the contract.
- **Compression ratio is per-skill.** Some skills are already dense and compress little; some verbose skills compress dramatically. Report a distribution, not just a mean.

## Sources

- Paper: *SkillZip: Evaluation-Free Skill Compression for Self-Evolving Agents by Discovering Reusable Structure* — Bai, Lin, Liu, Zhang, Jin, Cao, Li, arXiv 2608.11079, 2026.
