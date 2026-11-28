# Linear attention

*Taxonomy — attention / recurrent architectures that replace the O(N²) softmax scan with an O(N) state update.*

**TL;DR:** Linear-attention models trade a growing KV cache for a fixed-size recurrent state updated token-by-token. They inference faster and use less memory at long context, but they struggle with long-range associative recall and are historically unstable at scale. The modern defaults — GLA, Mamba-2, DeltaNet / GDN variants — differ mostly in how the forget gate and the write rule are structured.

**Related taxonomies:** [../architectures/_moe.md](../architectures/_moe.md)
**Depth files covered here:** [gated-deltanet](gated-deltanet.md) · [mla](mla.md)

---

## The problem

Softmax attention stores every past key-value pair (the KV cache) and scans all of them at each step. Memory and compute grow linearly with sequence length per token and quadratically over the whole sequence. At context lengths above ~128k this is the dominant cost of inference and often of training.

## The shared pattern

Compress the KV cache into a fixed-size matrix (or small set of matrices) S_t and update it recurrently:

S_t = F_t(S_{t-1}, k_t, v_t), where F_t is linear in S_{t-1}.

Inference becomes a pure RNN: constant memory, constant time per token. The design space is three choices — (1) how to *forget* old content (decay gate), (2) how to *write* new content (delta rule, outer product, selective), and (3) how to *read* (query projection into the state).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [gated-deltanet](gated-deltanet.md) | Delta-rule write with per-token forget gate; Stable GDN-2 adds symmetric erase | Plain GDN-2 unstable at scale (fixed by Stable variant) | Long-context inference, prefill-heavy workloads |
| Mamba / Mamba-2 | Selective state-space model with input-dependent A, B, C | Harder to parallelise during training than attention | Very long sequences (>1M tokens), causal-only |
| GLA (Gated Linear Attention) | Element-wise forget gate on outer-product write | Weaker associative recall than softmax | Mid-scale models where memory is the bottleneck |
| RWKV | Receptance-weighted time mixing, two-token updates | Older formulation; recent v6/v7 close most gaps | Pure RNN interpretability, edge deployment |
| [mla](mla.md) (hybrid) | Softmax attention with low-rank KV compression | Keeps softmax semantics; still O(N²) in FLOPs | When exact recall matters but KV memory is the bottleneck |

## How to choose

Default to **softmax attention + GQA** for anything under ~128k context — the training ecosystem is still much better. For long-context inference-first deployments, Mamba-2 and Gated DeltaNet are the serious contenders; Stable GDN-2 is the current "attention drop-in" story. MLA is not a linear-attention method but shares the "shrink the KV state" motivation and often stacks with linear attention in hybrid architectures. Hybrids (interleaving softmax and linear-attention layers) are increasingly common and often give the best of both.

## Adjacent but distinct

- Softmax attention variants (GQA, MQA, sliding-window): still O(N²) FLOPs; they shrink the KV cache but not the compute.
- Retrieval / external memory: offloads old content outside the model rather than compressing it in-model.
- Sparse attention (longformer, big-bird): keeps softmax but drops entries; neither compresses nor recurs.

## Sources

- Paper: Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability — Zhang et al., 2026 — [arXiv:2611.34051](https://arxiv.org/abs/2611.34051)
- Paper: Mamba-2 — Dao & Gu, 2024
- Paper: Gated Linear Attention Transformers with Hardware-Efficient Training — Yang et al., 2024
