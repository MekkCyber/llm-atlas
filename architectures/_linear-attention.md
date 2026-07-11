# Linear Attention

*Taxonomy — recurrent-memory alternatives to softmax attention with linear cost in sequence length.*

**TL;DR:** Softmax attention is $O(L^2)$ in sequence length. Linear-attention architectures replace the pairwise softmax with a **fixed-size recurrent memory** that is updated one token at a time — training runs in $O(L)$ and inference has constant-memory state per token. The family has proliferated (Mamba/SSMs, RWKV, RetNet, DeltaNet, Gated DeltaNet, Kimi Delta Attention, xLSTM); the modern default in hybrid stacks is a Gated DeltaNet or KDA layer interleaved with a small number of softmax attention layers.

**Related taxonomies:** [_normalization](_normalization.md) · [_moe](_moe.md)
**Depth files covered here:** *(none yet — populate as we write depth files for DeltaNet, Gated DeltaNet, Kimi Delta Attention, Mamba)*

---

## The problem

Softmax attention lets every token retrieve information from every earlier token, but the pairwise dot-product makes both training FLOPs and inference KV-cache memory scale as $O(L^2)$ and $O(L)$ per step. At long contexts — RAG traces, agentic rollouts, video frames — this is the dominant cost. Every technique in this taxonomy is trying to keep the *ability to route information across long distances* while paying a cost that stays linear in $L$.

## The shared pattern

Every variant expresses attention as a **recurrent memory** with (at least) three primitives:

1. **Write** — how the current token's key/value writes into a hidden state $H_t$.
2. **Erase / decay** — how the past state is decayed or selectively overwritten (this is where most of the variance lives).
3. **Read** — how the query retrieves from $H_t$.

Softmax attention is the "no compression, no decay" extreme: the memory stores every past token exactly. Linear-attention variants trade some retrieval fidelity for a memory that fits in a fixed-size state. The **delta rule** family (DeltaNet, Gated DeltaNet, KDA) writes with an outer-product update that subtracts the current prediction error — a memory-associative-array with continuous overwriting.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| DeltaNet (no depth file yet) | Delta-rule outer-product write on an associative-memory hidden state | Fixed-size state limits long-range recall | Long-context tasks where the tail matters less |
| Gated DeltaNet (no depth file yet) | Adds a learned gate on the erase step | Extra gate params; small throughput cost | Better memory retention than pure DeltaNet |
| Gated DeltaNet-2 (no depth file yet) | Refined gating with sharper decay control | Similar cost, sharper tuning | Language modeling at 350M–3B scale |
| Kimi Delta Attention (KDA) (no depth file yet) | Fine-grained per-dimension decay | Extra decay params | Reported lowest validation loss in the ETH 2026 sweep |
| Mamba / SSM (no depth file yet) | Selective state-space model; input-dependent A, B, C matrices | Different memory shape than delta rule | Speech, audio, non-language sequence modeling |
| RWKV (no depth file yet) | Time-mixing + channel-mixing recurrences | Older, simpler recurrence; less expressive | Small-model deployments |
| xLSTM (no depth file yet) | Modernized LSTM with exponential gating | Larger state per layer | Comeback line; competitive at small scale |

## How to choose

**Default in modern hybrid LLMs:** a mostly linear-attention stack with a few softmax layers interleaved (roughly 1 softmax layer per 3–7 linear layers) — the softmax layers preserve long-tail retrieval, the linear layers absorb throughput. Within the linear layers, **Gated DeltaNet** and **KDA** are the current front-runners; Mamba dominates non-language modalities.

**If maximum throughput matters:** pure Gated DeltaNet stacks with AdamW hit the best training throughput in the ETH 2026 sweep. Hybrid stacks close the loss gap but cost throughput.

**Optimizer:** **Muon** consistently lowered final validation loss vs AdamW across matched architectures in the delta family (ETH 2026 sweep). Worth trying.

**Cross-layer routing:** the Cross-Layer Value Routing (CLVR) trick — forward a lower layer's write-value into the aligned hidden stream of the next layer's memory — gives a small but consistent loss reduction on DeltaNet and Gated DeltaNet. Cheap to add.

## Adjacent but distinct

- **Sliding-window / sparse attention.** Still softmax; just cheaper to compute over local neighbours. Different tradeoff (drops distant tokens exactly vs compresses them lossily).
- **Attention variants** (MLA, GQA, MQA). Compress the KV cache but keep softmax semantics; complementary to linear attention, not a substitute.
- **State-space models (SSMs).** Overlap with this taxonomy — Mamba/S4 belong here — but the SSM community frames it as continuous-time dynamics, while the delta-rule family frames it as discrete recurrent associative memory. Same big idea, different heritage.

## Sources

- Paper: *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing* — Cerruti et al., ETH Zurich, 2026 — https://arxiv.org/abs/2607.07953 — the sweep this taxonomy summarizes.
- Paper: *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* — Yang et al., 2024 — DeltaNet.
- Paper: *Gated Delta Networks: Improving Mamba2 with Delta Rule* — 2024 — Gated DeltaNet.
- Paper: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu & Dao, 2023.
- Paper: *RWKV: Reinventing RNNs for the Transformer Era* — Peng et al., 2023.
