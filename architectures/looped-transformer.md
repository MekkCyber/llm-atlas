# Looped Transformer (Dory)
*Depth — a shared-weights recurrent transformer block that runs a variable number of iterations per request, with last-loop KV caching to keep serving memory flat.*

**TL;DR:** A single transformer block is applied **$L$ times with shared weights** instead of stacking $L$ independent blocks. One checkpoint serves many compute budgets: at inference, pick $L$ per request (or per token) based on difficulty. **Last-loop KV caching** writes only the final iteration's KV to cache, so a 3-loop pass has the same cache footprint as a 1-loop pass. Trained with a **difficulty-aware RL curriculum** that biases hard samples toward more loops. Dory1.0-4B (NVIDIA, 2026) at 2+ loops outperforms several larger baselines across 12 benchmarks while emitting fewer agentic tokens.

**Prereqs:** [transformer-block.md](transformer-block.md), [_normalization.md](_normalization.md)
**Related:** [qk-norm.md](qk-norm.md) · [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

Standard transformers pay parameters for depth: a 4B 32-layer model has 32 distinct transformer blocks. A looped transformer replaces this with **one** transformer block that is *applied* 32 times at inference — same compute, far fewer parameters. The idea is old (Universal Transformer, Dehghani 2018); what Dory contributes is a serving-friendly formulation plus an RL curriculum that uses the loop count as an adaptive compute knob.

---

## How it works

### Loop structure

Define a single transformer block $\mathcal{B}_\theta$ with shared parameters $\theta$. The $L$-looped forward pass is:

$$
h_0 = \text{embed}(x), \quad h_{\ell+1} = \mathcal{B}_\theta(h_\ell) \text{ for } \ell = 0 \ldots L-1, \quad y = \text{head}(h_L)
$$

Backbone is a **normalized transformer** (nGPT-style): all representations and weights are kept on the unit sphere, so iterating the same block $L$ times doesn't drift. Interleaved **sliding-window and full-attention** layers inside $\mathcal{B}_\theta$ balance local and global dependency capture.

### Last-loop KV caching

A naive loop writes $L$ copies of the KV cache per token. Dory writes **only** the final iteration's KV:

```
for ℓ in 0..L-1:
    if ℓ == L - 1:
        kv_cache[t] = compute_kv(h_ℓ[t])    # write only the last loop
    h_{ℓ+1} = B(h_ℓ)                         # all loops use intermediate KV in-flight
```

During decode, each new token reads the cached last-loop KV of previous tokens and runs its own $L$ loops over that fixed cache. Memory footprint matches a non-looped model.

### Difficulty-aware loop curriculum

During RLVR post-training, the loop budget is a hyperparameter per sample. The curriculum biases **hard samples** (low success rate at low $L$) toward **more loops** and keeps easy samples at low $L$. The model learns to spend compute adaptively — hard prompts get more iterations without explicit prompting.

At inference, loop count is picked per request. A classifier or heuristic estimates difficulty from the prompt; harder prompts get more loops.

---

## Why it matters

- **Parameter efficiency.** Dory1.0-4B at $L \ge 2$ beats several larger models on average across 12 benchmarks — effective depth without parameter cost.
- **Compute is a knob, not a fixed cost.** Serving stacks that today choose between a 4B and a 70B model can choose between $L=1$ and $L=8$ of the same model, with the cache cost of the 4B.
- **Fewer output tokens on agentic tasks.** More reasoning happens inside loops, less in the output. For long-horizon agents, this reduces downstream context pressure.
- **Last-loop KV caching is the practical unlock.** Previous looped-transformer work paid $L\times$ cache; Dory shows you don't have to.

---

## Gotchas & tricks

- **nGPT-style normalization is load-bearing.** Iterating a non-normalized block drifts representations off-manifold within a few loops. Unit-norm (or RMSNorm with careful scale) stabilizes iteration. Stripping normalization kills the method.
- **$L$ at training vs inference.** Training with a fixed $L$ fixes the serving $L$; training with variable $L$ (curriculum) lets you dial at inference. The paper trains with variable $L$ and shows the loop-count knob works post hoc.
- **Last-loop KV is approximate for very different $L$ values.** At inference time, serving tokens with $L=4$ against a cache written by $L=2$ prior tokens is a minor distribution mismatch. The paper reports it's negligible in practice; for very long contexts with large $L$ gaps, this is worth measuring.
- **Loop count scales latency linearly.** Unlike larger models, which can parallelize over devices, loops are serial. A 4-loop pass is ~4× the per-request latency of a 1-loop pass — pick $L$ based on request latency SLOs.
- **Difficulty estimation is upstream.** The "pick $L$ per request" step depends on a difficulty signal, which the paper provides at training time (RL verifier) but not at inference. Deployed systems need a cheap classifier.

---

## Sources

- Paper: *Dory1.0-4B: A Looped Normalized-GPT* — Sun, Acharya, Ladhak, Chmiel, Akle Serano, Banner, Ginsburg, Puvvada (NVIDIA, inferred), 2026, arXiv 2611.03998 — primary source.
- Paper: *Universal Transformers* — Dehghani et al., 2018 — the original shared-weights recurrent transformer idea.
- Paper: *nGPT: Normalized Transformer with Representation Learning on the Hypersphere* — Loshchilov et al., 2024 — the normalized backbone Dory uses.
