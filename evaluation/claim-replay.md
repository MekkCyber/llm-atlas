# Claim-replay auditing for evaluation suites
*Depth — the missing "does this eval actually support the claim attached to it" layer.*

**TL;DR:** An eval artifact specifies a forward computation (task, scorer, metric); it does not necessarily license the *claim* someone reads off the metric, because the historical evidence and alternative semantics needed to *replay* the claim under different assumptions are usually not shipped with the code. A commit-bound census of Inspect Evals finds that **110 of 124** mechanically-eligible units cannot support deterministic claim-replay — they stop before inference for lack of required grounding.

**Prereqs:** [README.md](README.md)
**Related:** [../safety/README.md](../safety/README.md)

---

## What it is

A formal framework and audit methodology for evaluation suites. Given a frozen substrate `D` (the shipped code + data at a pinned commit), a grounded family `F` of alternative-yet-defensible semantics, and a claim query `q`, the framework identifies the set of `F` under which `q` is deterministically resolvable at `D`. Everything outside the identified set is a *typed stop* — not a "fail" but a diagnosis of why the claim cannot be replayed.

## How it works

Per evaluation unit:

1. **Extract the claim query** — the specific proposition users read off the metric ("model X beats model Y on task T", "score exceeds threshold").
2. **Enumerate the grounded family** — alternative but valid re-instantiations (prompt template variants, scorer thresholds, tie-breaking rules, filter application order).
3. **Attempt deterministic inference** — for each element of the family, does `D` contain what is needed to resolve `q` exactly?
4. **Emit disposition** — pass through with exact value / winner / order / pairwise relation, or *typed stop* naming what evidence is missing.

Stops are classified so auditors can see *which* piece of history / semantics is unshipped: missing decontamination log, missing tie-breaking convention, unpinned judge model, ambiguous filter order, etc.

## Why it matters

Safety and capability leaderboards are increasingly consumed as "this model can / cannot do X." If most units cannot support deterministic claim-replay from their shipped artifacts, then published numbers are less durable than assumed and cross-model comparisons rest on undocumented assumptions. The framework gives eval maintainers a checklist — ship what's needed for claim-replay, or accept that your metric licenses only a narrow claim.

## Gotchas & tricks

- The audit is per-commit; adding evidence later doesn't retroactively license past claims.
- "Grounded family" is opinionated — narrower families identify more claims but at the cost of hiding assumptions.
- Not a robustness statement: a claim can be deterministically replayable *and* fragile under a natural perturbation. Both matter.
- Useful complement to reproducibility statements: reproducibility says "same input → same output"; claim-replay says "same claim → same verdict under stated assumptions."

## Sources

- Paper: *What Does an Evaluation License? A Commit-Bound Census of Claim-Relative Inference in Inspect Evals* — Xi Qin, 2026 — [arXiv:2608.19269](https://arxiv.org/abs/2608.19269)
- Inspect Evals — UK AISI's open-source eval suite audited by the paper.
