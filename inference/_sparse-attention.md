# Sparse Attention (Inference)

*Taxonomy — cutting inference-time attention cost by reading only a subset of the KV cache.*

**TL;DR:** Full attention at inference reads the entire KV cache every decode step (O(N) per step, O(N²) at prefill). Sparse attention limits that read to a small subset of tokens. Variants differ in **who decides the subset** (fixed pattern vs. runtime scorer vs. the model itself) and **when the decision is made** (offline vs. per-prompt vs. per-step). The modern trend is *dynamic, input-adaptive* selection — either via a lightweight proxy scorer (extrinsic) or by asking the model to declare its own scope (intrinsic).

**Related taxonomies:** [../fundamentals/attention](../fundamentals/attention.md)
**Depth files covered here:** [declarative-attention](declarative-attention.md) · [sparse-prefilling](sparse-prefilling.md)

---

## The problem

Modern long-context LLMs put quadratic prefill cost and linear-per-step decode cost on the KV cache. Empirically, only a small fraction of tokens carry the mass at any given step — the rest are dead weight the engine still has to read. Sparse attention exists to skip that dead weight without changing the model's answer.

Failure modes any variant must dodge:

- **Wrong tokens dropped** → accuracy collapses on retrieval-heavy tasks.
- **Selector overhead** ≥ savings → net slowdown at short-to-medium contexts.
- **Poor generalization** — a pattern tuned on one workload breaks on another.

## The shared pattern

Every sparse-attention scheme picks a mask `M ⊆ {1..N}` per query and computes attention only over `M`:

$$
\text{softmax}(QK_M^\top / \sqrt{d}) V_M
$$

The variants differ in how `M` is chosen — fixed vs. computed, extrinsic vs. intrinsic, per-prompt vs. per-step vs. per-head.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Fixed-pattern (sliding window, block-sparse, Longformer, BigBird) | Hand-designed mask (local + global) | No adaptivity to content | Pretraining-time; short-range tasks |
| Offline profiling (StreamingLLM, H2O) | Keep an "attention sink" + recent tokens by static rule | Miss the rare middle-of-context match | Chat with heavy recency bias |
| Runtime proxy scoring (Quest, MInference, SnapKV, Vertical-Slash routing) | Lightweight scorer over the KV cache picks top-k per step | Scorer still O(N); accuracy sensitive to the coverage threshold | Long-context QA and retrieval-heavy prefill |
| [sparse-prefilling](sparse-prefilling.md) — CRISP (2026) | Structural-mass routing off Vertical-Slash + sink-aware threshold instead of cumulative-coverage | Removes JSD/KL overhead and O(N) noise; still uses a scorer | Long-context prefill with a large KV |
| [declarative-attention](declarative-attention.md) — Declarative Attention (2026) | Model declares `<global>`, `<focus>`, `<local>` scopes inside its CoT; engine parses like tool calls | Requires the model to know its needs; zero-shot loses a few pp | Long-form reasoning where the model can plan its reads |

## How to choose

- **Retrieval-heavy long context (needle-in-haystack, code, docs):** a dynamic proxy scorer (CRISP-class) is the safest default. Fixed patterns miss rare-but-critical matches.
- **Long chatty context with strong recency:** offline sinks + recent window (StreamingLLM/H2O) still competitive and cheap.
- **The model does long chain-of-thought and can plan its reads:** intrinsic (declarative) — the model already knows what it needs; the extrinsic scorer is a middleman.
- **Combine layers:** several deployments run a dynamic scorer on a subset of heads and dense elsewhere. The routing decision itself is a knob.
- **Watch the coverage threshold.** A strict cumulative-mass threshold accumulates O(N) noise at long context (this is CRISP's motivation) — prefer sink-aware or top-k selection.

## Adjacent but distinct

- **Model-side sparse attention** ([architectures/](../architectures/)) — architectural variants (MQA, GQA, MLA, linear attention). Changes what's stored, not what's read at inference.
- **Speculative decoding** — orthogonal: reduces steps rather than per-step work.
- **KV cache quantization / eviction** — reduces per-token cost or removes tokens permanently; sparse attention keeps them but skips reading them.
- **Prefix caching** — reuse of KV across requests; independent of the sparse-read decision.

## Sources

- *Longformer* — Beltagy et al., 2020 — canonical fixed sliding + global.
- *StreamingLLM: Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023.
- *H2O: Heavy-Hitter Oracle* — Zhang et al., 2023 — offline attention-sink profiling.
- *Quest* — Tang et al., 2024 — query-aware KV selection.
- *MInference* — Jiang et al., 2024 — Vertical-Slash / A-shape / Block-Sparse patterns.
- *CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling* — Nguyen et al., 2026, [arXiv:2609.01925](https://arxiv.org/abs/2609.01925).
- *Language Models Can Control Their Own Attention* — Ho et al., 2026, [arXiv:2609.02737](https://arxiv.org/abs/2609.02737).
