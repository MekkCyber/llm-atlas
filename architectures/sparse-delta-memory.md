# Sparse Delta Memory (SDM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Gated linear RNNs (Mamba, DeltaNet, RWKV, GLA) get their compute advantage from a **fixed-size hidden state**, but that same fixed state caps long-context recall. Sparse Delta Memory scales the state by **orders of magnitude** without paying the FLOP bill: it treats the state as an addressable memory of $N_{\text{slots}}$ slots and updates only a **learned top-$k$ sparse subset per token** using the delta rule. Compute per token stays $O(k \cdot d)$ instead of $O(N_{\text{slots}} \cdot d)$. Introduced by Meta FAIR (2026).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [mla.md](./mla.md) · [../inference/README.md](../inference/README.md)

---

## What it is

Linear-attention / linear-RNN models fold the whole context into a **single fixed-shape state** $S \in \mathbb{R}^{d \times d}$ (or similar). Per token they perform an $O(d^2)$ update. This is cheap and streaming — but the state is small, and long-context recall (needle-in-haystack, associative retrieval) suffers compared to softmax attention.

The obvious escape — grow the state — kills the compute advantage: $O(d^2) \to O(N d^2)$ scales linearly in slot count.

SDM's escape: keep the total state big, but only touch a small **sparse subset** of slots per token, chosen by a learned key. The state becomes a **sparse associative memory** structurally close to a retrieval-augmented KV cache — but folded *inside* the RNN update rather than bolted on outside.

## How it works

**Slot memory.** State is $M \in \mathbb{R}^{N_{\text{slots}} \times d}$ with $N_{\text{slots}}$ orders of magnitude larger than a Mamba-style state. Each slot has its own key $K_j$ (learned, static or hashed).

**Sparse addressing.** For token $t$, compute a query $q_t$ from the input, then select the top-$k$ slots by score $\langle q_t, K_j \rangle$:
$$\mathcal{I}_t = \text{TopK}_j (\langle q_t, K_j \rangle), \quad |\mathcal{I}_t| = k$$

**Delta update on the selected slots.** For each $j \in \mathcal{I}_t$, apply the standard delta rule (from DeltaNet):
$$M_j \leftarrow M_j + \beta_t (v_t - M_j \cdot k_t) k_t^\top$$
Slots outside $\mathcal{I}_t$ are untouched — the update is fully sparse.

**Read.** Standard delta-rule read from the top-$k$ slots aggregated. Cost per token: $O(k d)$ for scoring + $O(k d)$ for update — **independent of $N_{\text{slots}}$** once addressing is done.

**Addressing must be cheap.** Naive scoring is $O(N_{\text{slots}} d)$ per token, which defeats the purpose. SDM uses either a product-quantized index (Faiss-style) or a hierarchical routing tree so top-$k$ scoring is sublinear in $N_{\text{slots}}$.

## Why it matters

- **Recall matches softmax attention at linear-RNN cost.** At matched inference FLOPs, SDM closes the long-context recall gap that Mamba-line models have carried, on needle-in-haystack and associative-retrieval evals.
- **Orthogonal to the recurrence.** The sparse-addressing pattern rides on top of the delta rule; the same trick should transfer to gated linear attention (GLA), RWKV, and other linear-RNN recurrences.
- **KV-cache-free long context.** The state is fixed per model, not per sequence. There is no $O(T)$ growth. Serving cost is decoupled from context length in a way softmax transformers cannot match — a serious advantage for agentic memory workloads where contexts are enormous and per-step compute matters.
- **Structural bridge to RAG.** The sparse-addressed memory looks like a learned in-model retrieval index. It suggests a design space where the "external retrieval / internal state" distinction dissolves.

## Gotchas & tricks

- **Addressing quality is the bottleneck.** Bad top-$k$ scores hide information. Product-quantized indices need training to align keys with queries.
- **Slot key initialization.** Random keys give near-uniform scoring; the model needs to learn distinct keys per slot. Author uses orthogonal init + explicit repulsion loss on keys.
- **Hard vs soft top-$k$.** Hard top-$k$ is non-differentiable in the slot indices. SDM approximates with a straight-through gradient or a softmax-relaxed top-$k$ during training.
- **Update sparsity vs capacity.** Small $k$ (e.g. 8) is FLOP-cheap but harder to write enough information per token; large $k$ approaches dense delta cost.
- **State size doesn't help decode-only tasks.** SDM's win is recall; on tasks with short context and heavy short-term dynamics, plain Mamba is fine.

## Sources

- Paper: *Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity* — Mazaré, Szilvasy, Douze, Lomeli, Auzina, Carpentier, Synnaeve, Jégou (Meta FAIR, Inria, U. Tübingen), 2026 — arXiv:2607.07386.
- Related: *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* — Yang et al., 2024 — the DeltaNet baseline.
- Related: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu, Dao, 2023.
