# Block-Diffusion Language Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training objective for language models that denoises **blocks of tokens jointly** rather than predicting one token at a time. The model autoregresses at the **block** level — block `k` conditions on blocks `0…k-1` — but within a block, all positions are predicted in parallel via iterative diffusion refinement. Trades some per-token perplexity for massive **parallel decoding speedups**: a block of `B` tokens takes `T` diffusion steps rather than `B` AR steps, and `T ≪ B` in practice. Scaled to MoE frontier size in Clover-35B-A3B.

**Prereqs:** [attention](../fundamentals/attention.md), [transformer-block](transformer-block.md)
**Related:** [gated-delta-networks](gated-delta-networks.md)

---

## What it is

Standard autoregressive LMs factor `p(x) = Π p(x_t | x_{<t})` and decode one token per forward pass.

Block-diffusion LMs partition the sequence into blocks of size `B` and factor:

```
p(x) = Π_k p(block_k | block_<k)
```

Each `p(block_k | block_<k)` is a **diffusion process** that starts from noise/masked tokens and iteratively refines to clean text over `T` denoising steps, conditioned on all prior blocks.

So **across blocks** it's autoregressive (usual causality), but **within a block** every position is predicted in parallel at every diffusion step.

---

## How it works

**Noising process.** Given a clean block `x_0`, define a sequence of increasingly-noised versions `x_1, x_2, …, x_T` by either (a) progressively masking tokens or (b) perturbing the categorical distribution over vocabulary. The forward (noising) direction is fixed and known.

**Denoising model.** Train a transformer `f_θ` to predict `x_0` from `x_t` and the prior clean blocks:

```
L = E_{x, t} [ CE( f_θ(x_t | block_<k, t),  x_0 ) ]
```

The loss is summed over diffusion timesteps `t` and over blocks.

**Attention structure.** The model uses **causal attention across blocks** (so block `k` cannot see block `k+1`) but **non-causal attention within a block** (all positions in block `k` can see each other during denoising). This is the key architectural requirement — the sequence mixer inside the block must tolerate bidirectional context, unlike standard decoder-only transformers.

**Sampling.** To generate, run `T` denoising steps per block, starting from fully-noised tokens and conditioning on already-generated blocks. One block costs `T` forward passes; a sequence of `K` blocks costs `K × T` passes instead of `K × B` AR steps.

**Decision-model trick.** A diffusion LM can fill multiple answer spans **jointly** in one block, letting correlations between related answers raise accuracy. AR models would require heavy prompting or multiple forward passes.

---

## Why it matters

Throughput: when `T ≈ 4–8` and `B ≈ 32–128`, inference is 4–30× fewer forward passes than AR for the same emitted tokens. Combined with MoE sparsity, this can hit real-time throughput on hard tasks.

Joint prediction: tasks with structured, correlated outputs (code with multiple related edits, forms with dependent fields, grouped Q&A) benefit from parallel denoising in ways AR cannot naturally replicate.

Clover-35B-A3B is the first open result where this trade-off produces a competitive agentic-coding score at frontier-adjacent scale — evidence that diffusion LMs are moving past perplexity-only comparisons.

---

## Gotchas & tricks

- **Within-block attention must be non-causal.** Causal mixers (vanilla decoder blocks) don't work. Use bidirectional softmax attention or a mixer family like [Gated Delta Networks](gated-delta-networks.md) that supports non-causal variants.
- **Block size is the critical knob.** Larger `B` means more parallelism per step but harder denoising; the quality / throughput frontier runs along `B`.
- **Long-horizon planning is harder than AR.** Block-level autoregression gives coarse control; failures tend to be whole-block semantic errors rather than off-by-one drift.
- **MoE routing is per-block, not per-token.** One routing decision serves the whole block during denoising, saving expert-dispatch overhead.

---

## Sources

- Paper: *Clover-35B-A3B-Preview Technical Report* — Chen et al., DiffuSpace, 2026 — [arXiv:2611.24636](https://arxiv.org/abs/2611.24636) — first frontier-adjacent open deployment.
- Earlier work: block-diffusion LMs appear in several 2024–2026 research lines (SEDD, Block-Diffusion papers); Clover is the first to demonstrate the recipe at this scale and on a hard agentic eval.
