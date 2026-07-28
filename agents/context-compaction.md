# Validated Context Compaction
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A context-management primitive that replaces a range of prior turns with a compacted summary, but only after *validating* that the summary preserves recall for the queries a downstream turn is likely to make. Turns the standard quadratic vs. lossy-linear tradeoff into a linear-cost, fidelity-preserving option — the operating point Maximem Synap reports 92% LongMemEval and 93.2% LoCoMo under.

**Prereqs:** [_context-management.md](./_context-management.md)
**Related:** [../inference/kv-cache-retrieval.md](../inference/kv-cache-retrieval.md)

---

## What it is

Naive context accumulation is **O(N²)** in conversation length (every turn re-attends over the growing prefix). Crude summarization is **O(N)** but sacrifices fidelity — the "summarization cliff" where later turns can't recover facts the summary dropped. Validated compaction closes that gap by treating the compaction step as a *replaceable* transformation whose output must clear a downstream-recall check before it's committed to memory.

## How it works

1. **Chunk selection.** Identify a range of turns eligible for compaction — usually older turns whose retrieval frequency has dropped.
2. **Compact.** Produce a structured summary (facts, decisions, entities, provenance pointers) using an LLM, keeping identifiers and links to the raw store so anything dropped can be re-fetched.
3. **Validate.** Run a battery of probe queries derived from the chunk itself against the summary. If recall on the probes falls below a threshold, either widen the summary or leave the chunk uncompacted.
4. **Commit + provenance.** Replace the compacted turns in the active window with the summary, but retain pointers to the raw turns and a provenance record of what was replaced when.

The validation step is what keeps the primitive from being "summarize and pray." The claim is that with validation gating, per-turn cost stays linear *and* recall stays measurably close to the un-compacted baseline.

## Why it matters

Long-horizon agents (support, research assistants, coding copilots) hit the compaction question every session. Without a validation gate, teams either accept quadratic bills or accept slow, silent recall decay. Framing compaction as a validated, provenance-preserving step turns it into an operational primitive with a knob (validation threshold) instead of a one-way risk.

## Gotchas & tricks

- **Probe generation is the hard part.** Probes must span the queries a downstream turn will actually make; naive random probes over-optimistically pass.
- **Provenance is non-optional.** If the summary drops a fact and no pointer to the raw turn survives, recovery costs a full re-load of the conversation.
- **Compaction ≠ eviction.** The compacted turns should typically stay in cold storage, not be deleted; recovery from a probe miss depends on it.
- **Interaction with KV caches.** Any prefix-cache benefit of the un-compacted prefix is lost on the turn compaction lands; batch compaction rather than compacting every turn.

## Sources

- Paper: *Agentic Context Management: Solving Agent Memory and Cost by Treating Them as Lifecycle and Architecture Problems* — Maximem, 2026 — [arXiv:2607.21503](https://arxiv.org/abs/2607.21503). Introduces the primitive and reports 92% LongMemEval / 93.2% LoCoMo under this configuration.
