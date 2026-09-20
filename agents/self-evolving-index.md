# Self-Evolving Search Index
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Traditional retrieval builds an index once and freezes its key representations. **SELF-INDEX** turns index construction into an online loop: an **Optimizer** diagnoses retrieval gaps from live traffic and rewrites index keys, while a **Query Simulator** proactively probes the index with synthetic queries to expose weaknesses before real users do. Retriever-agnostic; consistent improvements over static-index baselines across corpora and retrievers, with wins carrying into agent memory systems.

**Prereqs:** [README.md](README.md), [../data/_data-curation.md](../data/_data-curation.md)
**Related:** [coding-agent-harness.md](coding-agent-harness.md)

---

## What it is

An **index** is the compact set of representations a retriever searches over — chunk embeddings, keyword projections, symbolic keys. In classical IR, those representations come from a one-shot build step: choose a chunking scheme, encode chunks, ship. Retrieval failures propagate silently — the retriever ranks the wrong chunks; the user rewrites; the index never changes.

SELF-INDEX closes that loop. Two agents run continuously alongside the retriever:

- **Optimizer** — observes retrieval outcomes (which queries missed, which chunks were re-ranked or ignored downstream), diagnoses the miss (mis-chunking, wrong keys, missing paraphrases), and edits the index keys directly.
- **Query Simulator** — generates additional demands the current index cannot serve yet, driving proactive coverage rather than reactive patching.

The mechanism is retriever-agnostic — plug into a BM25, dense, or hybrid retriever.

## How it works

The evolution loop:

1. **Observe.** For each user query, log which chunks were retrieved, which were used downstream, and whether the answer succeeded. Signals include downstream verifier score, agent trajectory success, or explicit user feedback.
2. **Diagnose.** The Optimizer inspects miss patterns per query cluster — is the miss a chunking issue (the right content is in the corpus but split), a key-representation issue (the chunk embeds far from typical query phrasings), or a coverage issue (no chunk covers the concept)?
3. **Revise.** For a chunking miss, re-chunk the offending document; for a key miss, add paraphrased or expanded keys; for a coverage miss, escalate to the Query Simulator.
4. **Simulate.** The Query Simulator proposes plausible future queries against the current index. Queries that no chunk answers well are handed back to the Optimizer as coverage gaps.

All edits are additive or replace-in-place — the index never has a downtime rebuild.

## Why it matters

- **Retrieval-augmented agents** live and die by their index quality; a stale index is the silent common failure of long-lived agent memories.
- **Agent memory** specifically benefits — memory indexes drift as tasks evolve; self-evolution keeps them useful.
- **No retraining** — improvements come from index edits, not model updates. Cheaper and safer to deploy.

Reported gains: consistent improvements over static-index baselines across corpora and retrievers, with the same lift observed for search agents and agent-memory systems.

## Gotchas & tricks

- **Simulator drift.** Query simulators trained on the same corpus can hallucinate demands that never occur; ground simulation in real query logs when available.
- **Additive edits pile up.** Prune stale paraphrased keys periodically or the index bloats.
- **Diagnosis needs downstream signal.** Retrieval-hit rate alone is a weak feedback loop; wire in downstream verifier or agent-trajectory success.

## Sources

- "Self-Evolving Search Index" — Lee et al., 2026 — [arXiv:2609.19656](https://arxiv.org/abs/2609.19656).
