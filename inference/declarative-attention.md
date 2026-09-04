# Declarative Attention

*Depth — intrinsic sparse attention where the model declares its own read scope inside the chain-of-thought.*

**TL;DR:** Language models already know which parts of their KV cache they need to attend to for the next output — extrinsic proxy scorers are re-deriving that information. **Declarative Attention (DA)** (Ho et al., 2026) elicits the model to *declare* its scope in-line via three tags — `<global>` (full context), `<focus>` (a specific region), `<local>` (recent output only). The inference engine parses these like tool calls and skips most of the KV read. Zero-shot on Gemma-4-31B / Qwen-3.6-27B: **52.0% / 31.1% reduction in attended tokens during decoding** at only **1.27 / 2.75 pp** accuracy loss across 15 long-context benchmarks.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md), [_sparse-attention](_sparse-attention.md)
**Related:** [sparse-prefilling](sparse-prefilling.md)

---

## What it is

A protocol layered on the model's chain-of-thought that partitions each generation segment into one of three attention modes, chosen by the model itself:

| Tag | Scope | When |
| --- | --- | --- |
| `<global>` | Full KV cache | Ambiguous need; new topic; global summary |
| `<focus>` | A named region (e.g. "the JSON schema at position 12k") | Model knows where the relevant span lives |
| `<local>` | Only the recent decoded tokens | Continuing a local computation / rewriting |

The inference engine is modified to parse the tags mid-stream and route the *next* generated tokens to the corresponding sparse (or dense) attention path. It is *intrinsic* sparsity — the sparsity decision comes from the model, not from a scorer over KV activations.

## How it works

**Training-free protocol.** The paper's zero-shot setting uses a system prompt that teaches the model to bracket its output with the three tags. No fine-tuning is required to get the reported numbers.

**Engine integration.** The serving stack watches for the opening tag as tokens are emitted. On `<global>`, attention proceeds as usual. On `<focus>`, the model also declares the region (e.g. by summarizing which section it needs) and the engine restricts KV reads to that span. On `<local>`, only the newly generated tokens participate in attention. On the closing tag, mode resets.

**Two savings axes:**

1. **KV bytes read per step** — the dominant memory-bandwidth cost. Cut in direct proportion to the scope fraction.
2. **Attention FLOPs per step** — cut on the same axis, useful when the KV lives on-device.

Because the decision is made by the model as part of decoding, no auxiliary scorer runs and there is no O(N) per-step overhead — the sparsity budget is essentially free.

## Why it matters

- **Removes the scorer middleman.** Every dynamic sparse-attention method to date computes some proxy score over the KV cache to pick tokens; that score is still O(N). DA sidesteps the loop entirely by letting the model announce its intent.
- **Composes with everything else.** DA is orthogonal to KV quantization, paged attention, continuous batching, and speculative decoding — it's a *decision* about what to read, not a change to storage or scheduling.
- **Scales up gracefully.** The accuracy gap shrinks with model size (Qwen-3.6-27B loses only 2.75 pp), suggesting fine-tuned or RL-trained variants would close it entirely. Opens a new axis for sparse attention beyond scorer design.
- **Native fit for long-CoT models.** R1-style traces already plan globally then compute locally — DA's protocol maps onto that pattern almost verbatim.

## Gotchas & tricks

- **Zero-shot is imperfect at 30B.** A model that misdeclares its scope silently drops accuracy on retrieval-heavy segments. Watchdog: fall back to `<global>` when the declared `<focus>` region is small AND the task is a long-context QA class.
- **Focus regions must be nameable.** DA needs a way for the model to identify a region without re-scanning the KV. Structural anchors (section titles, code function names, JSON keys) work; opaque numeric offsets do not — teach the prompt to prefer named anchors.
- **Local mode is trickiest.** Prematurely dropping into `<local>` while a global dependency is still open (e.g. "the constant defined 40k tokens back") is the main failure mode. In zero-shot, cap consecutive local spans.
- **The engine has to be modified.** Unlike a fixed pattern, this requires mid-stream parsing and mode switching in the serving loop — not a drop-in flag in vLLM/SGLang until the runtime supports it natively.
- **A future training-based variant is the obvious next step.** RL with a reward that combines task accuracy and a mild `-|attended tokens|` penalty is the paper's suggested direction.

## Sources

- Paper: *Language Models Can Control Their Own Attention* — Namgyu Ho, Huzama Ahmad, Woosung Koh, Se-Young Yun, Tal Schuster, Cicero Nogueira dos Santos — 2026 — [arXiv:2609.02737](https://arxiv.org/abs/2609.02737) — KAIST · Google.
