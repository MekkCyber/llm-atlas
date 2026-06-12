# Visual Token Routing (Reroute)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** VLM visual-token pruners (FastV, PDrop, Nüwa-variants) follow a **rank-and-remove** paradigm: score tokens, keep top-K, *permanently discard* the rest. But token importance changes across decoder depth — tokens ranked low at layer 5 may be crucial at layer 25 for grounding queries. **Reroute** (Yang, Lo, Liu, 2026) replaces removal with **recoverable routing**: deferred tokens *bypass* the current decoder stage but stay alive, re-entering the candidate pool at the next routing decision. Theoretical TFLOPs and KV-cache budget stay in the same class as the baseline pruner; grounding accuracy under aggressive token reduction improves consistently.

**Prereqs:** [README.md](README.md)
**Related:** [../architectures/mla.md](../architectures/mla.md) · [../multimodal/README.md](../multimodal/README.md)

---

## What it is

VLMs project images into hundreds-to-thousands of visual tokens per image. Decoder inference becomes expensive in both attention compute and KV-cache memory. Visual-token pruning trims this — keep only the K most "important" tokens at each routing stage, drop the rest.

The standard mechanism is **rank-and-remove**: an importance score (attention to the text query, or a learned predictor) ranks tokens; the bottom drops out of subsequent computation forever. Once gone, a token can't contribute to deeper layers.

Reroute observes that importance is **layer-dependent**:
- Early layers tend to surface high-level scene information.
- Mid layers attend to query-relevant regions.
- Late layers re-engage with fine details for grounding.

A token that's "unimportant" at layer 5 (no current query-relevance) may carry a detail that becomes critical for the grounded answer at layer 25. Removing it early forecloses that path.

---

## How it works

### Defer instead of delete

At each routing stage $i$, the existing pruner (FastV, PDrop, etc.) produces a ranking. Reroute splits the candidate pool into:

- **Active set $S_i$**: top-K tokens by the pruner's score — pass through this decoder block normally.
- **Deferred set $D_i$**: tokens not in $S_i$ — **bypass** this block (no attention, no KV addition) but stay in the candidate pool for stage $i+1$.

At stage $i+1$, the ranker re-scores the full candidate pool — both the tokens that were active and the tokens that were deferred — using the current decoder state. Tokens that were deferred at $i$ can rejoin the active set at $i+1$ if they now look important.

### TFLOPs and KV-cache budget preserved

Bypassing tokens means they accumulate no KV-cache entries while deferred and consume no attention compute at that block. So the per-stage compute and KV-cache budget match the baseline pruner — Reroute keeps the same theoretical budget class.

The only added cost is the routing decision itself, which is dominated by the ranking computation the baseline pruner already does. Practical overhead is ≪ 1% per stage.

### Composes with existing pruners

Reroute reuses the pruner's existing scoring rule and stage-wise schedule. It's a structural change ("defer ≠ delete"), not a new scoring algorithm. Applied as a drop-in plug-in over FastV, PDrop, and Nüwa-variant pipelines on LLaVA-1.5 and Qwen backbones.

---

## Why it matters

- **Improves grounding under aggressive reduction.** Tokens crucial for grounding queries (which object, where) are the ones most prone to being killed early — they may not look important at layer 5 but matter at the grounding head. Reroute keeps them alive without exceeding budget.
- **Training-free.** No retraining. Drop into any deployed VLM with a token-pruning pipeline.
- **Same compute envelope.** Production token-pruning was chosen precisely for its compute savings; Reroute keeps the same envelope, just spends it more flexibly.
- **Generalizes the mechanism.** "Defer-vs-delete" reframes a whole class of dropping techniques (KV-cache eviction, layer skipping, expert routing) where the deletion may have been overhasty.

---

## Gotchas & tricks

- **Re-ranking cost compounds with depth.** Each stage re-scores the candidate pool; over many routing stages this adds up. For VLMs with many MoE-like routing checkpoints, batch the scoring.
- **Bypassed tokens have stale embeddings.** A deferred token has the decoder-block-$i$ representation; if it rejoins at $i+1$, its representation hasn't been updated by block $i$'s residual stream. The paper sweeps strategies for this; the simplest (use the pre-stage representation) works.
- **Improves grounding specifically, not general VQA.** Open-ended VQA queries don't always need late-layer fine-grained tokens; grounding (referring expressions, "where is X") does. Headline win is on grounding metrics.
- **Doesn't replace KV-cache compression.** Reroute is orthogonal to MLA-style KV compression (see [../architectures/mla.md](../architectures/mla.md)) — one reduces *which* tokens have KV entries, the other reduces *how big* each entry is. Stack both.

---

## Sources

- Paper: *Reroute, Don't Remove: Recoverable Visual Token Routing for Vision-Language Models* — Yang, Lo, Liu (NYCU / NTU), 2026 — [arXiv 2606.12412](https://arxiv.org/abs/2606.12412).
- Paper: *FastV* and *PDrop* — baseline rank-and-remove pruners Reroute plugs into.
