# Governed Memory

*Depth — tagging each agent-memory entry with quality / confidence / lifecycle / conflict metadata, then retrieving for precision rather than top-K cosine.*

**TL;DR:** Standard agent memory is "embed snippet, append, retrieve by cosine, top-K." **Governed memory** upgrades this to a structured store where every entry carries explicit metadata: a quality score, a confidence level, a lifecycle stage (active / stale / superseded), a verifier outcome, and conflict signals against existing entries. Retrieval becomes *precision-oriented* — return fewer, higher-confidence entries rather than top-K cosine. Introduced as the KnowledgeBank upgrade in GeneralVLA-2 (Wang et al., 2026), evaluated on agent-coding benchmarks beyond robotics.

**Prereqs:** [_agent-memory](_agent-memory.md)
**Related:** [memory-governance](memory-governance.md), [hierarchical-agent-memory](hierarchical-agent-memory.md)

---

## What it is

Per-entry metadata schema (the exact field set varies; the principle is what matters):

| Field | Purpose |
| --- | --- |
| **Quality** | A learned or human-assigned score of the entry's usefulness. |
| **Confidence** | The agent's belief that the entry is correct. Updated by verifier outcomes. |
| **Lifecycle** | `active` / `stale` / `superseded`. Drives expiry and replacement. |
| **Verifier outcome** | Result of running a check against the entry (did the tool actually work, did the fact match a source). |
| **Conflict signals** | Pointers to other entries that contradict this one, with the conflict mode (factual / preference / outcome). |

Retrieval policy is **precision-first**: filter entries by lifecycle = `active` and confidence ≥ threshold *first*, then rank the survivors by relevance. Returns fewer items; what's returned is higher quality.

## How it works

Write path:

```
new candidate entry
  → compute initial quality + confidence
  → check against existing entries
      → if conflict detected, mark both entries with the conflict pointer
      → resolve via verifier or downgrade lower-confidence entry to `stale`
  → assign lifecycle = active, write to store
```

Read path:

```
query
  → filter by (lifecycle = active) AND (confidence ≥ θ) AND (relevance > 0)
  → rank survivors by relevance × confidence
  → return top-K (often K small: 1–3 high-confidence > top-10 cosine)
```

Periodic maintenance: a verifier re-runs over the store, downgrading stale entries and confirming or invalidating ambiguous ones.

## Why it matters

Two effects, both shown empirically in GeneralVLA-2:

- **Higher action success on long-running agent tasks.** Reported +4.53% on Terminal-Bench SR and +3.73% on SWE-Bench Verified resolve rate over ReasoningBank, while reducing "agent steps" by ~5%. The mechanism is straightforward: avoiding low-confidence retrievals avoids the failure cascade where the agent acts on bad memory and has to recover.
- **The machinery to actually implement [memory-governance](memory-governance.md).** Forgetting becomes lifecycle = `superseded`; access control attaches to the per-entry metadata; conflicts get explicit handling. Without governance metadata, those properties are heuristic at best.

## Gotchas & tricks

- **Confidence calibration is the hard part.** A miscalibrated confidence score (overconfident on bad entries) makes precision-oriented retrieval *worse* than cosine top-K. Use verifier outcomes to calibrate.
- **Stale ≠ delete.** Lifecycle = `stale` should be queryable for audit but invisible to normal retrieval. This matters for [memory-governance](memory-governance.md) — forgetting needs `deleted`/`tombstoned` as a separate state.
- **Conflict resolution is policy-dependent.** Factual conflicts can be resolved by verifier; preference conflicts often can't be (both entries are true at different times) and need lifecycle + recency rather than overwrite.
- **Composes with [hierarchical-agent-memory](hierarchical-agent-memory.md).** Hierarchical scopes *what* lives in which store; governance metadata *grades* every entry within a store. Most production stacks will want both.

## Sources

- Paper: *GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning* — Wang, Ma, Zhang, Guo, Shi, Tang, 2026 — https://arxiv.org/abs/2606.17480
- Reference baseline: *ReasoningBank* (referenced in the GeneralVLA-2 paper) — the prior cosine-retrieval baseline GeneralVLA-2 improves over.
