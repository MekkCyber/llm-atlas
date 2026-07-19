# KV-Cache Grafting
*Depth — persist a KV state as a byte-exact artifact and splice it into a fresh context on demand.*

**TL;DR:** Run a "study session" once, snapshot the resulting KV cache, and later restore that cache into a new inference context. Under a pinned deterministic transformer configuration, the grafted logits are bit-identical to a fresh full-context computation (SHA-256 equality). Frozen weights, no gradient step, no per-use retrieval prompt inflation — an alternative to fine-tuning and RAG for "adding knowledge" to a served model.

**Prereqs:** [../architectures/multi-head-attention](../architectures/multi-head-attention.md)
**Related:** *(no other inference depth files yet — this is the folder's first)*

---

## What it is

A transformer processes a prompt into a KV cache: for each attention layer, a `(K, V)` tensor pair per token that later decode steps re-read. The cache is normally scoped to one request.

KV-cache grafting reframes the cache as a **first-class artifact**: persist `(K, V)` for a whole document (a book, a codebase, an internal wiki), then load it as the *prefix* of a fresh inference request. Subsequent decoding attends over the grafted prefix as if the model had just processed it.

Prefix caching does the same thing *within a live server process*. KV grafting extends the idea across processes, machines, and long time gaps by making the artifact portable and reproducible.

## How it works

The mechanism has two halves.

### Deposit

1. Pin the transformer's inference configuration (dtype, kernel choices, RNG seed, batch shape).
2. Run the prompt (the "study session") through the model to fill the KV cache.
3. Serialize `{K^ℓ, V^ℓ}` per layer with the configuration hash as key.

### Graft

1. Load the KV artifact into a fresh inference process using the same pinned configuration.
2. Continue attention from the grafted cache: new tokens attend over grafted K/V as if the model had just processed the deposit prompt.
3. Under bit-equal float ops, the resulting logits are byte-exact to a fresh recomputation on the concatenated prompt.

The "byte-exact" property is what separates grafting from lossy KV compression: SHA-256 of the logit vector matches; the token distribution has zero KL from a freshly-computed one; argmax agreement is 100% across many samples.

## Why it matters

- **Amortize study cost.** Read a 500K-token document once, save the KV, then answer many independent queries with 0 marginal prefill cost per query.
- **Avoids the RAG prompt-inflation tax.** RAG re-pastes the retrieved evidence into every request. Grafting installs it as prefix state, sharable across requests.
- **No fine-tuning.** Weights stay frozen — no gradient step, no forgetting, no per-domain checkpoint sprawl.
- **Determinism claim = knowledge claim.** Because the graft is bit-exact, the grafted model *provably* behaves as if it had processed the deposit prompt — this is what makes it a knowledge mechanism, not just a latency optimization.

## Gotchas & tricks

- **Configuration lock-in.** Bit-exactness holds only under the pinned config. Changing the attention kernel, dtype, or GPU generation breaks byte equality (though behavior stays very close).
- **KV artifact size.** A 500K-token cache at fp16 for a 12B model is tens of GB per layer summed. Compression helps but breaks the determinism claim — pick your side.
- **Multiple grafts per request.** Concatenating two independent deposits (book A + book B) does *not* generally match a joint-processing of both, because attention across the two grafts was never computed at deposit time. Use one canonical deposit or accept the divergence.
- **Positional encoding matters.** RoPE and friends encode absolute positions into K; grafting an artifact deposited at positions `[0, N)` into a fresh context also at `[0, N)` is fine, but shifting the graft into a later slot needs positional re-encoding, which breaks byte equality.
- **Not a substitute for RAG when the corpus updates.** Grafts are per-snapshot. If the underlying document changes, the artifact must be re-deposited.

## Sources

- Paper: *Byte-Exact KV-State Grafting Turns a Frozen Small Model into a Verified-Knowledge Flywheel* — Sietse Schelpe, 2026 — the deterministic deposit-and-graft mechanism and the SHA-256 equality result.
- Related: prefix caching in production serving stacks (vLLM, SGLang) as the in-process precursor.
