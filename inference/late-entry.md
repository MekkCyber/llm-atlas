# Late Entry (for Looped Language Models)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An inference-time speedup specific to [looped language models](../architectures/looped-language-models.md): for newly appended tokens running against an already-computed deep-cached context, run only the **last K of T** loop iterations instead of all T. A lightweight verifier (**Wake**) decides per-token whether late entry is safe; where it isn't, fall back to full T passes. Combined with light fine-tuning it beats early-exit at matched compute and gives greedy-exact speedups of 5–8× on Ouro-2.6B and Huginn-3.5B at T=64.

**Prereqs:** [looped-language-models.md](../architectures/looped-language-models.md), [attention](../fundamentals/attention.md)
**Related:** [inference README](README.md)

---

## What it is

Looped LMs iterate a shared stack of layers T times per token. When generating from a long cached prefix, every cached token has already been through all T loops. A *new* token starts with no loop history. Early-exit addresses easy tokens by stopping *early* in the loop. Late-entry does the opposite: it **starts the new token's loops at pass T−K+1**, reusing the fact that the deep context already carries almost all the information.

The remaining risk is tokens where the first few passes materially change the representation. A **verifier** is trained to flag those tokens and reroute them to full T passes.

---

## How it works

For a new token $x_t$ in a cached context:

1. Initialize $h^{(T-K)}_t$ from the embedding (or a shallow projection), skipping passes $1 \ldots T-K$.
2. Run passes $T-K+1, \ldots, T$ with attention over the fully-processed cached context.
3. Apply **Wake**: a cheap per-token head that predicts whether the late-entry output differs from the full-pass output. If above threshold, redo the token with all T passes.

Wake is trained with a small amount of supervision: compare greedy-exact next-token prediction with and without late entry on a held-out set, and classify disagreements.

**Hyperparameters.** T=64 (as in Ouro / Huginn); K is swept per model; typical K = 4–16.

---

## Why it matters

- **Greedy-exact speedup.** Unlike early-exit, where quality degrades gradually, late-entry + Wake produces token-by-token identical outputs to full T-pass inference at the configured K. 5.2× on Ouro-2.6B and 8.4× on Huginn-3.5B at T=64.
- **Larger lossy speedups.** Raising the Wake threshold trades accuracy for throughput; reported lossy speedups up to 12.5×.
- **Looped LMs become serveable.** T=64 at full compute is prohibitive for interactive serving. Late-entry closes that gap without quality loss, making the parameter-efficiency of looped LMs reach production.

---

## Gotchas & tricks

- **Only works on cached context.** Late entry relies on the deep context already having been looped T times. The first prompt still needs full T passes to build the cache — this is a decode-time speedup, not a prefill speedup.
- **Verifier quality bounds the win.** A too-permissive Wake threshold silently degrades quality; too strict negates the speedup. Tune on held-out generations with the target task.
- **Not compatible with early-exit at the same time.** Early-exit and late-entry assume opposite things about where the "important" compute lives in the loop. Picking one per deployment is the sane default.
- **Fine-tuning required.** Zero-shot late-entry hurts quality even at large K; a small LoRA-scale fine-tune of the last K passes recovers greedy-exact behavior.

---

## Sources

- Paper: *Late Entry in Looped Language Models* — 2026 — [arXiv:2611.24576](https://arxiv.org/abs/2611.24576). Introduces late entry and the Wake verifier.
- Related background: *Universal Transformers* (Dehghani 2018) and the Ouro / Huginn open-weight looped reasoning models.
