# Subquadratic Architectures

*Taxonomy — recurrent / state-space alternatives to softmax attention with sub-quadratic sequence-length scaling.*

**TL;DR:** Softmax attention is $O(L^2)$ in sequence length. A family of architectures replaces it with **recurrent-style state updates** whose cost is $O(L)$: Mamba/Mamba-2 (selective state-space models), xLSTM (revived LSTMs with exponential gating + matrix memory), Gated DeltaNet (delta-rule fast-weights), RWKV (gated linear attention), and linear-attention variants. They all expose a fixed-size hidden state that's updated per token, and they differ in **what that state can track over long sequences** — finite-state automata, accumulators, key-value associations. A 2026 head-to-head (Zólyomi et al., JKU Linz) cast them in a common memory-dynamics framework and showed xLSTM uniquely handles both **finite-state tracking and accumulation** simultaneously.

**Related taxonomies:** none yet — sibling to attention variants but mechanistically a different family.
**Depth files covered here:** *(no depth files yet; this taxonomy seeds them)*.

---

## The problem

Attention is the workhorse of transformer LMs but scales as $O(L^2)$ in sequence length for both time and memory. At long contexts (32k, 128k, 1M tokens), the quadratic cost dominates inference and training. Two paths address it:

1. **Sparse / approximate attention** — keep softmax attention but compute only a subset (sliding window, sparse patterns, low-rank approximations). Still attention-shaped.
2. **Subquadratic architectures** — replace softmax attention with a recurrent or linear-attention mechanism whose per-token cost is $O(1)$ in sequence length and total cost $O(L)$.

This taxonomy covers path 2: structurally non-attention architectures.

---

## The shared pattern

Every subquadratic architecture in this family has:

- A **fixed-size hidden state** $h_t$ that's updated per token.
- An **update rule** $h_t = f(h_{t-1}, x_t)$ that's recurrent in time.
- An **output rule** $y_t = g(h_t, x_t)$ producing the next-token logits / activation.

The state can be a vector (classic RNN), a matrix (associative memory, fast weights), or a structured operator (state-space matrices). The update rule determines what the architecture can *track* and what it *forgets*.

Cast in a unified recurrent-matrix-update form, each variant's expressive power can be analyzed mechanically — what finite automata it can simulate, what counts it can accumulate, what key-value associations it can recover.

---

## Variants

| Technique | Update rule shape | What its state tracks | Main tradeoff |
| --- | --- | --- | --- |
| **Mamba-2** (Dao & Gu, 2024) — *no depth file yet* | Selective SSM: data-dependent $\Delta$, $A$, $B$, $C$ over a continuous-time state | Linear ODE state with input-selective forgetting | Strong on language modeling; weaker on state-tracking benchmarks |
| **xLSTM** (Beck et al., 2024) — *no depth file yet* | Exponential gating + matrix memory (mLSTM) + scalar memory (sLSTM) | Both accumulation and finite-state tracking | Most flexible; harder to parallelize than SSMs |
| **Gated DeltaNet** (Yang et al., 2024) — *no depth file yet* | Delta-rule fast-weight update on a matrix state | Key-value associations | Fast retrieval; weak on long-range counting |
| **RWKV** (Peng et al., 2023) — *no depth file yet* | Time-decayed linear attention with gating | Exponentially-decayed key-value | Simple, efficient; limited state-tracking expressivity |
| **Linear Attention** (Katharopoulos et al., 2020) — *no depth file yet* | $\phi(q) \cdot \sum_t \phi(k_t) v_t^T$ over a matrix memory | Linear key-value associations | Cheapest; weakest expressivity baseline |

---

## How to choose

**Default for long-context language modeling (2026):** there isn't one yet. The frontier has not settled — hybrid attention + subquadratic stacks (Mamba/attention interleaved) currently win at scale, suggesting no single subquadratic architecture is a strict transformer replacement.

**If you need finite-state tracking and counting:** xLSTM. The 2026 Zólyomi et al. analysis shows it uniquely handles both simultaneously, and empirically wins on code and time-series benchmarks where these capabilities matter.

**If you optimize purely for language-modeling perplexity at scale:** Mamba-2 has the strongest LM scaling results, and its hardware-aware kernel is the most mature.

**If you need key-value retrieval (long-context recall):** Gated DeltaNet's delta-rule update is the most explicit associative-memory mechanism among these.

**If you want maximum simplicity:** RWKV or linear attention. Weaker expressivity but easiest to integrate.

**Modern frontier models hybridize:** Llama 4, Jamba, and Composer 2-class systems interleave attention layers with SSM/Mamba layers — letting attention handle in-context recall while SSM layers handle the long context cheaply. See [../case-studies/composer2.md](../case-studies/composer2.md).

---

## Adjacent but distinct

- **Sparse / sliding-window attention** — still softmax attention, just with restricted patterns. Different family; covered under attention variants.
- **Linear attention from kernelized softmax** — sits on the boundary; included above as the simplest variant of this family.
- **Mixture-of-Experts** — sparse activation across *experts*, not sub-quadratic in sequence length. Solves a different cost problem; see [_moe.md](_moe.md).
- **KV-cache compression** ([mla.md](mla.md), paged attention) — keeps softmax attention but shrinks the cache. Orthogonal to subquadratic.

---

## Sources

- Paper: *On Subquadratic Architectures: From Applications to Principles* — Zólyomi, Stap, Hoedt, Schmidinger, Hauzenberger, Böck, Klambauer, Hochreiter (JKU Linz / ELLIS), 2026 — [arXiv 2606.12364](https://arxiv.org/abs/2606.12364) — unified memory-dynamics framework + head-to-head benchmark.
- Paper: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu & Dao, 2023.
- Paper: *Mamba-2: Transformers are SSMs* — Dao & Gu, 2024.
- Paper: *xLSTM: Extended Long Short-Term Memory* — Beck et al., 2024.
- Paper: *Gated DeltaNet* — Yang et al., 2024.
- Paper: *RWKV: Reinventing RNNs for the Transformer Era* — Peng et al., 2023.
- Paper: *Transformers are RNNs (Linear Attention)* — Katharopoulos et al., 2020.
