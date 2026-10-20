# Speculative Decoding
*Taxonomy — drafter families for accelerating autoregressive LLM inference.*

**TL;DR:** Speculative decoding generates multiple tokens per target-model forward pass by proposing them with a cheap **drafter** and **verifying** them against the target. The variants differ in *what the drafter is* (independent LM, head on the target, state reader) and *what it reads* (its own KV, the target's KV, the target's hidden states or recurrent state). Modern recipes split along two axes: **drafter form** (external vs embedded) and **history source** (own KV vs target state).

**Related taxonomies:** _(none yet)_
**Depth files covered here:** [history-free-drafter](history-free-drafter.md) · [hybrid-la-speculative-drafter](hybrid-la-speculative-drafter.md)

---

## The problem

Autoregressive LLM decoding is memory-bound: each token forces a full forward pass that reads the whole KV cache. On a GB-class target, that forward pass is dominated by KV reads, not FLOPs. If a cheap predictor can propose $k$ plausible next tokens, the target can **verify them all in one forward pass** (parallel softmax over $k$ positions). Accepted tokens commit; the first rejected token forces a correction and the loop restarts. Expected speedup is roughly the average accepted length per draft, capped by the target's own entropy.

## The shared pattern

Every speculative decoder:

1. **Drafts** $k$ candidate tokens cheaply.
2. **Verifies** them with one target forward pass, computing accept/reject per position under a provably-correct sampling rule (importance-sampled or greedy).
3. **Commits** the accepted prefix and corrects at the first rejection.

What changes across variants is how the drafter is built and what it uses as context.

## Variants

| Technique | Drafter form | History the drafter uses | Main tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| **Vanilla (external LM)** | Separate small LM | Its own KV cache | Easy to swap in; drifts from target on long/evolving contexts | Short single-document generation on static context |
| **Medusa** | Multiple extra heads on target | Target's final hidden state | No separate model; heads are shallow and local | Any softmax target; cheap, trainable alongside |
| **EAGLE / EAGLE-2** | Tiny transformer reading target hidden states | Target hidden states + its own small cache | Higher acceptance length; needs training alignment with target | Softmax targets at high batch size |
| **[DLite](history-free-drafter.md)** (history-free) | Light head reading target hidden state | **No own cache** — target state only | Flat memory per request; cannot look back beyond target state | Agentic workloads (long, mutating contexts) |
| **[ReaDy](hybrid-la-speculative-drafter.md)** | Learned query + state reader | Target's **recurrent state** | Specialized to hybrid linear-attention targets | Hybrid LA / Mamba-style targets |
| **DSpark** | External LM | Own KV + target-aware calibration | Simpler stack than hidden-state drafters | Softmax targets, no hybrid-LA need |

## How to choose

- **Softmax transformer target, static context (document generation, batch inference)** → EAGLE-2 or Medusa. Both are mature; EAGLE-2 wins on acceptance length at higher cost.
- **Agent loop, context evolves across turns** → DLite ([history-free-drafter.md](history-free-drafter.md)). The lack of drafter state is a feature, not a cost.
- **Hybrid linear-attention or Mamba-style target** → ReaDy ([hybrid-la-speculative-drafter.md](hybrid-la-speculative-drafter.md)). Vanilla and head-on-target drafters assume full KVs they can read; those don't exist here.
- **Low concurrency, latency-critical** → any drafter. The speedup is largest at low batch sizes where decoding is memory-bound.
- **Very high concurrency** → check whether the drafter's own memory overhead degrades throughput; history-free drafters shine here.

## Adjacent but distinct

- **Parallel decoding (Blockwise, Lookahead)** — generate several tokens in one target forward without a separate drafter, using Jacobi or Gauss-Seidel iteration. Simpler, but acceptance is much lower.
- **Draft-and-verify with retrieval** — the drafter conditions on retrieved snippets rather than running an LM. Niche for RAG workloads.
- **Non-autoregressive decoding** — predict all positions at once; a different design space entirely.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — the original.
- Paper: *Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE / EAGLE-2* — Li et al., 2024.
- Paper: *DLite* — Wang et al., 2026 — [arXiv:2610.23787](https://arxiv.org/abs/2610.23787).
- Paper: *ReaDy* — Sun et al., 2026 — [arXiv:2610.20503](https://arxiv.org/abs/2610.20503).

---

## Conventions

- **Filename:** `_speculative-decoding.md` (leading underscore — taxonomy).
- **Folder placement:** `inference/` root, alongside the depth files it covers.
- **Scope:** drafter families and the pattern they share. Target-side mechanics (paged attention, prefix caching) live elsewhere.
