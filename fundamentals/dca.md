# Dual Chunk Attention (DCA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **training-free** long-context extension for RoPE-based LLMs. Instead of scaling position indices (PI, NTK, YaRN) or extending the base frequency (ABF), DCA keeps the trained RoPE table untouched and **rewrites the relative-position matrix at inference time** so every $(q, k)$ pair sees a relative offset that lies inside the pretrained range. Achieved by splitting the sequence into chunks of size $s$ (typically $s = \tfrac{3}{4} c$ where $c$ is the pretrained context length) and using **three different position-remapping rules** depending on whether the $(q, k)$ pair is **intra-chunk**, **successive-chunk**, or **inter-chunk**. Extends Llama 2 (4K pretrained) to 32K with negligible perplexity drift; to 96K–192K with moderate drift; integrates with FlashAttention; **composes with PI / NTK / YaRN**. Introduced by An et al., 2024 (arXiv 2402.17463, "ChunkLlama"). Used in Qwen 2.5 at inference time to push dense models from 32K → 128K and Turbo to 1M tokens.

**Prereqs:** [attention.md](./attention.md), [rope.md](./rope.md)
**Related:** [_positional-encoding.md](./_positional-encoding.md) · [multi-head-attention.md](./../architectures/multi-head-attention.md) · [qwen2-5.md](./../case-studies/qwen2-5.md)

---

## What it is

Three structural choices that distinguish DCA from every other RoPE-extension method:

1. **Where the modification happens** — at inference time, in the attention kernel. No weights change. No fine-tuning. No new positions are ever passed to RoPE.
2. **What is modified** — the *position index used for each* $(q, k)$ *pair*, not the position-encoding function or the base frequency.
3. **How the position is chosen** — three different rules conditional on whether the pair is in the same chunk, an adjacent chunk, or a distant chunk.

Compare to the alternatives, which all touch RoPE itself:

| Method | What changes | Needs fine-tune? |
| --- | --- | --- |
| Position Interpolation (PI) | Scale position indices down by $s = L'/L$ | Yes (short adapter) |
| NTK-aware scaling | Scale the base frequency $b \to b \cdot s^{d/(d-2)}$ | Optional |
| YaRN | Wavelength-aware NTK + attention temperature | Yes (short adapter, fewer tokens) |
| ABF | Just use a larger base ($b = 500\mathrm{K}$ or $1\mathrm{M}$) | Pretrain choice |
| **DCA** | **Remap relative position index per** $(q,k)$ **pair, inside the pretrained range** | **No** |

DCA is the only method in this family that **doesn't change what RoPE sees** — it changes *which RoPE positions are paired with which tokens*. That is why it's training-free.

---

## How it works

### The chunking

Split a sequence of length $L'$ into chunks of size $s$. Each token's chunk index is $\lfloor i / s \rfloor$.

Typical setting from the paper:
- Pretrained context length: $c = 4096$ (Llama 2)
- Chunk size: $s = 3072 = \tfrac{3}{4} c$
- Local-window size: $w = c - s = 1024$

For models with longer pretrained context (Together-32K, CodeLlama-16K), $s$ scales with $c$ — they use $s = 24{,}000$ for $c = 16{,}000$ or $32{,}000$, extending to 192K total context.

### Three position-remapping rules

Given a query at position $i$ and a key at position $j$, define the chunk distance $d_{\text{chunk}} = \lfloor i/s \rfloor - \lfloor j/s \rfloor$. DCA assigns one of three remapped relative offsets:

**Intra-chunk** ($d_{\text{chunk}} = 0$). Same chunk. Reset positions modulo $s$:

$
P_q^{\text{Intra}}[i] = i \bmod s, \qquad P_k[j] = j \bmod s
$

$
M[i][j] = P_q^{\text{Intra}}[i] - P_k[j] \quad \text{(local relative position, } < s\text{)}
$

Within a chunk, tokens see normal relative positions in $[0, s)$ — strictly inside the trained range. RoPE applies exactly as it did during pretraining.

**Successive-chunk** ($d_{\text{chunk}} = 1$). Adjacent chunks. The query uses a sliding-window-style position, then saturates:

$
P_q^{\text{Succ}}[i] = \begin{cases} s + (i \bmod s) & \text{if } i \bmod s < w \\ c - 1 & \text{otherwise} \end{cases}
$

For the first $w$ tokens of a chunk, $P_q^{\text{Succ}}[i]$ smoothly continues from where the previous chunk ended; after that, it pins to $c-1$. This preserves locality at the chunk boundary (the most positionally-sensitive region) while keeping every remapped position $\le c - 1$.

**Inter-chunk** ($d_{\text{chunk}} > 1$). Distant chunks. All queries see the maximum pretrained position:

$
P_q^{\text{Inter}}[i] = c - 1
$

$
M[i][j] = (c - 1) - P_k[j] \ge c - s
$

Far-apart pairs are bucketed: any token attending to one a chunk-or-more away sees the same "far" relative offset. The model can still distinguish them by content, just not by precise distance. This is the structural cost of DCA — long-distance positional resolution is sacrificed to keep all remapped positions in range.

### Assembly

The final relative-position matrix is piecewise:

$
M[i][j] = \begin{cases} P_q^{\text{Intra}}[i] - P_k[j] & \text{if } d_{\text{chunk}} = 0 \\ P_q^{\text{Succ}}[i] - P_k[j] & \text{if } d_{\text{chunk}} = 1 \\ P_q^{\text{Inter}}[i] - P_k[j] & \text{if } d_{\text{chunk}} > 1 \end{cases}
$

The attention score is then standard scaled dot-product with RoPE applied via $M[i][j]$ instead of the raw $i - j$:

$
\mathrm{attn}(q_i, k_j) = \mathrm{softmax}\!\left( \frac{q_i \cdot R(M[i][j]) \cdot k_j}{\sqrt{d_h}} \right)
$

where $R(\cdot)$ is the same RoPE rotation matrix used during pretraining. No new positions, no scaling factors, no adapter.

### Why it's training-free

Every $M[i][j]$ produced by the three rules satisfies $0 \le M[i][j] < c$ — strictly inside the range RoPE saw during pretraining. The model's learned weights have already seen every angle $R(m)$ for $m < c$; DCA just chooses which angles to apply to which pairs.

By contrast: PI/NTK/YaRN ask the model to interpret RoPE at *new* angles (scaled-down positions, larger base frequency) that didn't appear in pretraining. A short fine-tune is the usual fix. DCA sidesteps the problem by refusing to produce new angles in the first place.

### Composition with PI / NTK / YaRN

DCA is **orthogonal** to base-frequency or position-scaling methods. The paper extends:
- **CodeLlama** (NTK-aware, $c = 16\mathrm{K}$) → 192K with DCA on top
- **Together-32K** (PI-trained, $c = 32\mathrm{K}$) → 192K with DCA on top

The composition: the base model uses its own (rescaled / extended) RoPE definition, DCA remaps the position indices within that RoPE table at inference. Qwen 2.5 chains these: ABF + YaRN + DCA to go from 4K pretraining to 128K serving on dense models.

### FlashAttention compatibility

DCA's per-pair $M[i][j]$ can be precomputed as three position vectors ($P_q^{\text{Intra}}$, $P_q^{\text{Succ}}$, $P_q^{\text{Inter}}$) and passed into the attention kernel. The paper provides a FlashAttention-2-compatible implementation. GPU memory and throughput "comparable to original self-attention in Llama" per the paper.

This is a real win — ReRoPE (a competing training-free method) cannot use FlashAttention and OOMs at 16K context.

---

## Empirical highlights

**Perplexity drift on PG19 (Llama-2-7B with DCA, "ChunkLlama2"):**

| Context | 4K | 8K | 16K | 32K | 65K |
| --- | --- | --- | --- | --- | --- |
| PPL | 7.87 | 7.67 | 7.64 | 7.89 | 15.87 |

Essentially flat to 32K (8× extension), degrades sharply past that.

**Llama-2-70B + DCA goes further:**

| Context | 4K | 8K | 16K | 32K | 96K | 192K |
| --- | --- | --- | --- | --- | --- | --- |
| PPL | 5.24 | 5.18 | 5.21 | 5.30 | 5.80 | 7.05 |

24× extension (4K → 96K) at +0.56 PPL. The bigger the base model, the more headroom DCA has.

**Vs. competing training-free methods at 32K (PG19, Llama-2-7B):**

| Method | PPL @ 32K |
| --- | --- |
| Llama-2 (vanilla RoPE) | $> 100$ (fails) |
| Llama-2-PI | 15.11 |
| Llama-2-NTK | 58.91 |
| Llama-2-NTK-YaRN | 11.74 |
| **Llama-2 + DCA** | **7.89** |

**Vs. fine-tuned baselines at 32K:**

| Model | Training | PPL @ 32K |
| --- | --- | --- |
| LongLoRA-32K 7B | Yes | 7.80 |
| CodeLlama-16K 7B | Yes | 8.36 |
| **Llama-2-7B + DCA** | **No** | **7.89** |

Matches fine-tuned models without any training.

**End-task: ChunkLlama2-Chat-70B reaches 94% of GPT-3.5-16K average on L-Eval zero-shot** (63.20 vs 67.03 averaged across TOEFL/QuALITY/Coursera/SFiction).

**Passkey retrieval:** 100% accuracy up to 18K with vanilla DCA on 13B; 90% accuracy at 192K with Together-32K + DCA.

---

## Why it matters

- **First training-free method to match fine-tuned long-context models** on perplexity and downstream benchmarks. Closes the "I have a pretrained model and want longer context tomorrow" gap completely for many use cases.
- **Composes with everything.** PI, NTK, YaRN, ABF, FlashAttention — DCA sits on top of them. Qwen 2.5's 4K → 128K (and Turbo → 1M) pipeline relies on this composability.
- **No new training compute.** Compare to YaRN's "10× fewer tokens than PI" — DCA's training compute is zero. The cost is paid at inference, and is negligible.
- **The trick is structural, not parametric.** DCA reframes long-context extension as a *position-bucketing* problem: how many distinct relative offsets does the model need to distinguish? For most tasks, the answer is "fewer than the context length" — same chunk, adjacent chunk, or far away covers most of what matters.

---

## Gotchas & tricks

- **Loses long-distance positional resolution.** All inter-chunk pairs (chunk-distance $> 1$) collapse to relative offset $c-1$. The model can tell "same chunk" from "different chunk" cleanly, but can't tell "5 chunks away" from "20 chunks away" by position alone — only by content. For tasks requiring precise long-distance position (counting tokens at far distances), DCA may underperform a fine-tuned long-context model.
- **Chunk size** $s$ **is load-bearing.** The paper's $s = \tfrac{3}{4} c$ is empirical. Too small → many cross-chunk pairs → more pairs collapse to the same far-offset, hurting quality. Too large ($s = c$) → no margin for the successive-chunk window, locality at chunk boundaries breaks. Stay near $\tfrac{3}{4}$.
- **Successive-chunk window** $w = c - s$**.** The "smooth handoff" at chunk boundaries depends on this. If you change $s$, $w$ changes; don't tune them independently.
- **PPL degrades past \~24× extension.** Llama-2-7B + DCA at 65K shows PPL 15.87 (vs 7.87 at 4K). The method is not unlimited. Beyond about $24c$, you need a longer pretrained base or to combine with ABF/YaRN.
- **Small models extend less than big models.** 7B caps around 32K; 70B reaches 96K cleanly. Capability headroom scales with base-model size, same as every long-context method.
- **Position bias at sequence start.** Passkey-retrieval ablations show ChunkLlama's first accuracy drop appears near the beginning of the document (opposite of NTK, which fails near the middle). Different from "lost in the middle" but real — check failure mode on your task.
- **Doesn't change KV-cache size.** Each token still produces one $(K, V)$ entry. DCA modifies attention positions, not the cache layout. KV memory still scales as $O(L')$ with the extended length.
- **Implementation note.** The cleanest implementation is to precompute three position vectors ($P_q^{\text{Intra}}$, $P_q^{\text{Succ}}$, $P_q^{\text{Inter}}$) per layer and select between them with a chunk-distance mask before the attention kernel. The paper provides a "monkey patch" against HuggingFace's `LlamaAttention`.
- **Don't combine with sliding-window-only attention naively.** DCA *is* a structured form of sliding-window-plus-global; layering another sliding-window mask on top can over-restrict attention and tank quality. If you want sliding-window for compute reasons, design the masks together.
- **Used in production.** Qwen 2.5 dense models (7B–72B) ship DCA + YaRN at inference for 128K context. Qwen 2.5-Turbo extends this to 1M with an additional sparse-attention mechanism on top. The "training-free" claim survives at frontier scale.

---

## Sources

- Paper: *Training-Free Long-Context Scaling of Large Language Models* — An, Ma, Lin, Chen, Yuan, Kong, 2024, [arXiv 2402.17463](https://arxiv.org/abs/2402.17463) — the original DCA / ChunkLlama paper. Section 3 (method), Section 4 (perplexity), Section 5 (passkey), Section 6 (L-Eval).
- Paper: *Qwen2.5 Technical Report* — Qwen Team / Alibaba, 2024, [arXiv 2412.15115](https://arxiv.org/abs/2412.15115) — applies DCA + YaRN as the inference-time long-context stack for all dense Qwen2.5 models, scaling 32K → 128K.
- Paper: *YaRN: Efficient Context Window Extension of Large Language Models* — Peng et al., 2023, [arXiv 2309.00071](https://arxiv.org/abs/2309.00071) — the RoPE-extension method DCA composes with most often.
- Paper: *Extending Context Window of Large Language Models via Positional Interpolation* — Chen et al., 2023, [arXiv 2306.15595](https://arxiv.org/abs/2306.15595) — PI, also composable with DCA.
- Code: [HKUNLP/ChunkLlama](https://github.com/HKUNLP/ChunkLlama) — the official DCA implementation with FlashAttention-2 patches.
