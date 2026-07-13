# Linear Attention

*Taxonomy — attention variants with linear (rather than quadratic) cost in sequence length, based on maintaining a bounded-size recurrent state.*

**TL;DR:** Softmax attention costs $O(L^2)$ in sequence length; **linear attention** replaces the softmax with a kernel or gating scheme that lets attention be rewritten as a **recurrent update on a fixed-size state**, making both training and inference $O(L)$. Modern variants (DeltaNet, Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2, Mamba, RWKV) all fit a **generalized delta-rule** template — they differ in how the recurrent state is updated and gated. The modern deployment pattern is **hybrid**: keep a few softmax layers (for recall) and route the rest through gated delta-rule layers (for throughput at long context).

**Related taxonomies:** [_moe.md](_moe.md) · [_normalization.md](_normalization.md)
**Depth files covered here:** [delta-rule](delta-rule.md) · [multi-head-attention](multi-head-attention.md) · [mla](mla.md)

---

## The problem

Softmax attention over $L$ tokens materializes an $L \times L$ score matrix. Training and prefill are $O(L^2)$ compute and $O(L^2)$ memory (with FlashAttention, memory drops to $O(L)$ but compute is still quadratic). At 128K+ context this dominates cost.

Two structurally different fixes exist. (1) **Cheaper softmax attention** — smaller K/V (MQA, GQA), compressed K/V (MLA), sparse patterns. (2) **Non-softmax attention** — replace the softmax with something that lets attention run as a $O(L)$ recurrence. This taxonomy covers path (2).

## The shared pattern

Every linear-attention variant is a **recurrence on a bounded state matrix** $S_t$:

$$
S_t = f(S_{t-1}, k_t, v_t), \qquad o_t = g(S_t, q_t)
$$

The state $S_t$ is a $d_k \times d_v$ matrix (fixed size, independent of $t$), $f$ is an update rule, $g$ is a readout. Because $S_t$ has fixed size, both training and inference are $O(L)$.

The **delta rule** family (DeltaNet and descendants) parameterizes $f$ as a Krotov/Hopfield-style associative update: given a key–value pair $(k_t, v_t)$, replace the state's readout for $k_t$ with a blend of the old readout and $v_t$. Modern variants add **gating** (input, output, or both) so the model can selectively write to the state — this is what makes them competitive with softmax attention on hard recall.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Linear attention (Katharopoulos et al., no depth file yet) | Replace softmax with a feature map $\phi$; state = $\sum \phi(k)v^\top$ | Weak recall; degrades vs softmax | Original formulation; largely superseded |
| [DeltaNet](delta-rule.md) | State updated by a Krotov/Hopfield-style delta write | Better recall than plain linear; still no gates | Baseline for the delta-rule family |
| Gated DeltaNet (GDN, no depth file yet) | Add per-token gating to the delta update | More expressive; slightly more compute per step | Recall-heavy long-context tasks |
| Kimi Delta Attention (KDA, no depth file yet) | Variant of gated delta with production-tested gates from Kimi-K1.5 | Empirically strong; details ship inside Kimi | Frontier open-model deployment |
| Gated DeltaNet-2 (GDN-2, no depth file yet) | Refined gating and value-side updates over GDN | Best pure-linear result in the 2026 ETH study | Hybrid stacks or when softmax is unavailable |
| Mamba / SSM (no depth file yet) | Selective state-space model — same $O(L)$ regime, different math | Different implementation (parallel scan), similar tradeoffs | Long-context, structured sequences |
| RWKV / RWKV-7 (no depth file yet) | Time-mixing + channel-mixing with a WKV recurrence | Different recurrence; strong at small–medium scale | Efficient long-context; deployment-ready recurrent stacks |
| **Hybrid attention routing** (no depth file yet) | Cross-layer router mixes softmax and linear layers | Adds a routing decision to the stack design | The modern default — captures most of the linear-attention benefit without softmax's recall gap |

## How to choose

**Full-softmax stack** — still the safest default at model scales where cost is not the bottleneck.

**Hybrid** (softmax + gated linear) — the emerging default at long context. Keep a handful of softmax layers (usually the first few) and route the rest through a gated delta-rule variant. The ETH 2026 study finds the routing decision matters more than the specific linear variant.

**Pure linear** — attractive when you have to ship long-context on a memory budget that softmax can't meet, or in RNN-like deployment settings where the fixed-size state is a feature. Expect a recall gap on tasks that stress associative retrieval.

**Which linear variant** — if you can pick, Gated DeltaNet-2 or Kimi Delta Attention are the current strongest. Plain DeltaNet is a good baseline for research; plain linear attention (Katharopoulos) is mostly a reference point now.

## Adjacent but distinct

- [mla](mla.md) — MLA is *cheaper softmax*, not linear attention. Same $O(L)$ KV-cache memory story but the score matrix is still softmax and $O(L^2)$ compute.
- [multi-head-attention](multi-head-attention.md) — the softmax baseline that this taxonomy compares against.
- **Sparse attention** (sliding window, longformer, sparse routing) — reduces the cost of softmax attention by restricting which pairs are scored. Different lever than replacing softmax outright.

## Sources

- Paper: *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing* — Cerruti et al. (ETH Zurich), 2026 — [arXiv 2607.07953](https://arxiv.org/abs/2607.07953). The head-to-head study that motivates this taxonomy.
- Paper: *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* — Katharopoulos et al., 2020.
- Paper: *Parallelizing Linear Transformers with the Delta Rule Over Sequence Length* (DeltaNet) — Yang et al., 2024.
- Paper: *Gated Delta Networks* — Yang et al., 2024.
- Paper: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu & Dao, 2023.
- Paper: *RWKV: Reinventing RNNs for the Transformer Era* — Peng et al., 2023.
