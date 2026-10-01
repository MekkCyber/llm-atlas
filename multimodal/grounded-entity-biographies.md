# Grounded Entity Biographies (GEB)
*Depth — long-video memory indexed by physical entity rather than by time.*

**TL;DR:** Answering questions about hour- or day-long videos often means connecting events involving the **same physical object** across clips. Chronological descriptions and text-derived entities confuse different objects that share a label and split observations of a single object across event boundaries. **GEB** (Ren et al., UIUC / Microsoft, 2026) groups visually-grounded observations of the same physical instance into a *biography* — a retrievable memory keyed by entity, with per-moment context preserved.

**Prereqs:** [README.md](./README.md)
**Related:** [../agents/README.md](../agents/README.md), [../agents/context-language-models.md](../agents/context-language-models.md)

---

## What it is

Classical long-video RAG indexes clips by time or by caption-derived entities. Both fail predictably:

- **"Different objects, same description."** Two blue mugs are two different physical objects; a text index collapses them.
- **"Same object, disconnected observations."** A mug seen at 10:14 and 14:03 may have no text link between those moments; a time index doesn't connect them.

GEB resolves identity *visually* — using re-identification against grounded detections — and builds per-entity memory. Each physical instance gets a *biography*: an ordered list of observation records, each with its clip, timestamp, pose, caption, and context. Queries about an entity retrieve its biography; queries about a moment still retrieve by time; queries about an event still retrieve by action.

## How it works

Three stages:

1. **Grounded observation extraction.** For each clip, run a grounded detector/VLM that emits `(bbox, visual-feature, caption, context)` for each visible object. This produces a sea of per-moment entity observations.
2. **Instance linking.** Merge observations across clips using visual re-identification features (similarity above a threshold). The threshold is chosen so false merges stay rare — biographies are allowed to be sparse rather than wrong.
3. **Biography indexing.** For each entity, store an ordered observation list and a canonical visual fingerprint. Retrieval supports entity-conditioned queries ("show every time X was moved") alongside standard time/action queries.

Downstream VLMs/agents read the biography as context when a question is entity-centered. Per-moment context is preserved (bbox, timestamp, surrounding caption) so the retrieval is grounded in actual video evidence, not a textual summary.

## Why it matters

- **Hour- and day-scale assistants need entity memory.** Home cameras, meeting assistants, and long-form reasoning agents fail on object-persistent questions. GEB is a concrete way to turn visually-rich memory into retrievable structure.
- **Clean counterpart to text-RAG.** Where RAG retrieves relevant *text*, GEB retrieves relevant *instances* — a different index axis, complementary rather than competing.
- **Precondition for persistent embodied agents.** Robots and wearable assistants that operate for hours accumulate many observations of the same objects; without entity-indexed memory, they re-perceive instead of remember.

## Gotchas & tricks

- **Re-ID thresholds matter.** Too low → false merges across different objects collapse biographies into incoherence. Too high → every observation becomes its own singleton biography (back to no linking). The paper tunes conservatively.
- **Visual similarity ≠ identity.** Two instances of the same product model are visually near-identical. GEB can split them only when they coexist on screen or with other context cues; otherwise it keeps a single biography. This is often the right call for QA.
- **Occlusion and viewpoint change.** A biography can gap when the object is unobserved; downstream queries must tolerate gaps, not treat them as "object disappeared."
- **Memory grows.** Biographies for thousands of objects over hours of video are large. A GC policy (drop short biographies, drop stale ones) is part of a practical deployment.

## Sources

- Paper: *Beyond the Timeline: Augmenting Long-Video Memory with Grounded Entity Biographies* — Ren, Fan, Pao, Guo, Zia, Chen, Schwing, Hua — UIUC / Microsoft, 2026 — [arXiv:2609.38155](https://arxiv.org/abs/2609.38155).
