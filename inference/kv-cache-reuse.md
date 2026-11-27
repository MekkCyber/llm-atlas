# KV-Cache Reuse Across Context Edits
*Depth — keep the valid prefix of the KV cache after an in-place edit of the LLM's context, re-prefilling only from the edit point onward.*

**TL;DR:** When an LLM agent edits its own context mid-task — compacting a span, pinning or dropping a region — the standard implementation re-prefills the entire context after the edit, wasting prefill compute on the unchanged prefix. **KV-cache reuse** keeps the cache for everything before the edit and recomputes only from the edit point onward. The NVIDIA 2026 study on model-managed context reports up to **15% prefill savings per request** at no measurable accuracy loss when this is implemented cleanly.

**Prereqs:** [README](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../agents/context-management.md](../agents/context-management.md) · [../fundamentals/rope.md](../fundamentals/rope.md)

---

## What it is

Prefill is the forward pass over the input context that populates the KV cache before decoding begins. For long contexts (32K+ tokens, common in agent harnesses), prefill dominates per-request compute.

Every context edit in an agentic loop — "compact these old tool calls into a summary," "drop this region," "insert a reminder here" — invalidates part of the KV cache. The naïve response is to re-prefill the whole edited context. KV-cache reuse observes that the *prefix* up to the edit point is still valid and does not need to be recomputed.

## How it works

Given context $C = C_{\text{prefix}} \oplus C_{\text{edited}} \oplus C_{\text{suffix}}$:

- The KV cache $K_{\text{prefix}}, V_{\text{prefix}}$ for the prefix is unchanged by the edit.
- The new context is $C' = C_{\text{prefix}} \oplus \text{edit} \oplus C_{\text{suffix}}$.
- Prefill only the edit $\oplus$ suffix. Reuse $K_{\text{prefix}}, V_{\text{prefix}}$ untouched.

Two compatibility conditions:

1. **Positions must align** across the valid prefix. If the edit changes the *length* of the pre-suffix region, every position in the suffix shifts and RoPE / sinusoidal embeddings baked into the prefix's K values now mismatch. Either preserve length (edit in place, same token count) or re-prefill the suffix too.
2. **Attention sinks and global tokens** that reach backward across the edit need their KV re-derived if they attend to the edited region. In practice, limit global tokens to positions *before* the edit or re-prefill them explicitly.

The clean pattern:

```python
# Before edit
prefix_kv = model.prefill(prefix_tokens)
full_kv   = prefix_kv + model.prefill(rest_tokens, kv=prefix_kv)

# Edit happens at the start of rest
new_rest  = edit_tokens + suffix_tokens
full_kv_2 = prefix_kv + model.prefill(new_rest, kv=prefix_kv)  # reuse prefix_kv unchanged
```

## Why it matters

- **Up to 15% prefill compute saved per request.** In the NVIDIA study, agent loops that edit context mid-task cut per-request prefill by this much while matching the no-reuse baseline on task accuracy.
- **Enables model-managed context in practice.** If every `compact` call re-prefills the whole context, the model-managed pattern ([../agents/context-management.md](../agents/context-management.md)) is too expensive to be useful. KV reuse is what makes it cheap enough to deploy.
- **Composes with paged attention.** Paged-attention serving systems (vLLM, SGLang) already track KV at page granularity; KV reuse across edits slots in as "which pages stay valid" bookkeeping.

## Gotchas & tricks

- **Position-shifting edits break reuse.** An insert that lengthens the pre-suffix region changes the positional encoding for every suffix token. Either keep the edit length-stable or accept full re-prefill.
- **Learned absolute position embeddings.** Models that use learned (non-rotary, non-sinusoidal) position embeddings are harder to reuse because position embeddings are mixed into the input; most modern LLMs use RoPE and are fine.
- **Numerical drift across backends.** If the prefix was prefilled on one backend (e.g. FlashAttention) and the suffix on another, FP8 rounding differences can accumulate. Pin both to the same kernel path.
- **Watch the attention mask.** Causal attention is fine — the prefix never looks at the edit. Bidirectional or windowed attention schemes may let the prefix attend back into the edit, invalidating the cache.

## Sources

- Paper: *Letting Models Manage Their Own Context: Where We Stand* — Devoto, Vem, Safavi, Kariyappa, Tseng, Furgale, Silveira Cabral (NVIDIA), 2026, arXiv 2611.34147 — introduces KV reuse across context edits in the agent-harness setting; reports the 15% prefill savings.
- Related: [../agents/context-management.md](../agents/context-management.md) — the agent-side pattern that generates the edits KV reuse exploits.
