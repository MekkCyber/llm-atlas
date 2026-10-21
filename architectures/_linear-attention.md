# Linear Attention

*Taxonomy — attention variants with sub-quadratic sequence complexity, driven by an associative matrix (or state-space) memory.*

**TL;DR:** Softmax attention is $O(S^2)$ in sequence length — fine for training, painful at long-context inference. Linear-attention variants replace the exact softmax with a recurrent summary over a fixed-size state, trading some recall capacity for $O(1)$ per-token memory and compute. All variants share the same core: *write* new (key, value) pairs into a running state at each step, *read* with the query. They differ in how the state is updated and how writes interact.

**Related taxonomies:** [_normalization](_normalization.md), [_moe](_moe.md)
**Depth files covered here:** [kronecker-delta-rule](kronecker-delta-rule.md) · [multi-head-attention](multi-head-attention.md) (contrast)

---

## The problem

Softmax attention stores every past token's K and V explicitly (the KV cache). At 128K-context serving, the KV cache dominates GPU memory and all-reduce cost. Any inference-time technique that caps state size at the hidden-dimension level — rather than growing with sequence length — unlocks much larger batches, much longer context, or much cheaper serving. The constraint every linear-attention variant fights is: **how much *recall* can you keep when the state budget is fixed?**

## The shared pattern

Every variant maintains a bounded-size state $S_t$ (a matrix, a state-space, or a small set of tensors) and runs two operations per token:

1. **Write.** Fold the new $(k_t, v_t)$ into $S_t$ via some update rule.
2. **Read.** Produce the output as a function of $q_t$ and $S_t$ (typically $S_t^\top q_t$).

The *write* rule is the knob. Pure additive writes (early linear transformers) suffer catastrophic interference as the state fills; delta-rule writes correct a residual; gated writes let the model forget; low-rank Kronecker writes avoid rewriting unrelated associations.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Linear Transformer (feature-map) | Associative state $S_t = \sum_i \phi(k_i) v_i^\top$ | No forgetting — state saturates | Short context, uniform recall |
| RWKV | Time-decay exponential moving average on state | Decay is parametric, hard to make selective | Long sequences with locality bias |
| Mamba / SSM | Selective state-space $S_t = A_t S_{t-1} + B_t x_t$ | Discretisation overhead; parallel scan complexity | Streaming, strong at copy/retrieval with selectivity |
| DeltaNet | Delta rule — correct full residual along $k_t$ | Full-vector overwrite causes interference | Associative recall on short-to-mid context |
| Gated DeltaNet / Gated DeltaNet-2 (ref'd by MemT-1.0) | DeltaNet + input / forget / write-target gates | More params per step; gate tuning sensitive | Longer context, memory-native agents |
| [kronecker-delta-rule](kronecker-delta-rule.md) (KronLSTM) | Rank-one Kronecker outer-product update | Scalar correction can under-shoot large residuals | Long-context recall, NIAH, real-world retrieval |

Link techniques with a depth file; leave others as plain text until a depth file lands.

## How to choose

- **Default for recall-heavy long context:** KronLSTM or Gated DeltaNet-2. The delta-rule family currently has the strongest NIAH scores among sub-quadratic architectures.
- **Default for streaming / audio / state-space fit:** Mamba (selective SSM). SSMs are the strongest when the input has strong temporal locality that the selection mechanism can latch onto.
- **Pure efficiency with low recall needs:** RWKV. The decay-only state is cheapest to train and serve.
- **Hybridisation wins.** Interleaving one softmax-attention block every 4–8 linear-attention blocks (Jamba, Zamba, several internal Mamba hybrids) closes the remaining long-tail recall gap at a small cost. Treat full attention as a scarce resource, not the default.

## Adjacent but distinct

- **[MLA](mla.md)** — a KV-cache compression that still runs *softmax* attention; shares the "shrink the state" motivation but is not sub-quadratic.
- **Sliding-window attention** — bounds the KV cache by truncation, not by a learned recurrent summary.
- **Flash-attention / paged attention** — kernel-level efficiency over the standard quadratic attention; orthogonal to linear-attention memory choices.

## Sources

- Survey: *Linear Attention: Progress, Challenges, and Scaling Laws* — various, 2024–2026.
- Foundational: *Transformers are RNNs* — Katharopoulos et al., 2020.
- *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu & Dao, 2023.
- *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* — Yang et al., 2024.
- *Kronecker Delta Rule for Long Context Linear Attention* — Zólyomi et al., 2026.
