# Hybrid Attention
*Taxonomy — transformer variants that mix full and linear/state-space attention under a fixed budget.*

**TL;DR:** Pure full-attention transformers pay $O(L^2)$ for context length; pure linear/SSM variants pay $O(L)$ but lose recall on long-range tasks. Hybrid attention keeps a **small number of full-attention layers** in an otherwise linear/SSM stack — the full layers carry precise long-range recall, the linear ones carry everything else. The family splits along two axes: *which* layers to keep full, and *how* to convert a pretrained dense transformer into the hybrid.

**Related taxonomies:** [_moe](_moe.md)
**Depth files covered here:** *(no depth files yet — populated as individual methods are added)*

---

## The problem

Three uncomfortable facts:

- Full attention is the only known mechanism that reliably preserves single-token recall at long context.
- Full attention dominates compute past ~32k tokens and dominates memory past ~128k.
- Linear-attention and SSM (Mamba, GLA, Gated DeltaNet) layers are cheap but lose on exact lookup and needle-in-haystack tasks.

Hybrid architectures ask: what's the minimum full-attention budget that preserves long-context recall?

## The shared pattern

```
block_i(x):
  if i in keep_full_set:
     x = full_attention(x)
  else:
     x = linear_or_ssm(x)
  x = FFN(x)
```

Two design knobs:

- **Layer selection**: which indices live in `keep_full_set` (typically 10–25% of layers).
- **Conversion recipe**: start from scratch with a hybrid, convert a pretrained dense model with distillation, or do neural architecture search over the layer set.

Converted models dominate when there is a strong pretrained dense model to recycle; from-scratch hybrids win when training budget is ample.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Zamba / Jamba (from scratch) — *no depth file yet* | Mamba backbone with a handful of full-attention layers from the start | Loses the pretrained dense checkpoint | Training a hybrid from scratch |
| MOHAWK (conversion via distillation) — *no depth file yet* | Replace FFN+attn blocks with SSM blocks, distill each stage | Multi-stage distillation is expensive | Converting a strong dense pretrain into a hybrid |
| Mamba-in-Llama — *no depth file yet* | Swap some attention layers for Mamba, lightweight distillation | Simple; layer selection is coarse | Quick-and-dirty conversion |
| FlashMorph (off-policy search) — *no depth file yet* | Score layers off-policy, pick top-K full-attention layers, retrain | Off-policy proxy diverges from recovery objective | Fast search, moderate quality |
| FlashMorph2 (on-policy) — *no depth file yet* | Sample hybrid architectures, evaluate end-to-end, jointly optimize layer-selection probabilities + weights under the recovery loss | Needs a budget for the inner sampling loop | Best reported quality under fixed GPU-hour budget |

## How to choose

**If converting a strong open-source dense model (default 2026):** FlashMorph2-style on-policy search. 0.72–3.57 GPU-hours across 0.6B–8B models with better recall than off-policy predecessors.

**If training from scratch and you don't need a dense baseline:** a Jamba-style hybrid with ~20% full-attention layers placed uniformly. Simple and robust.

**If deployment constraints are tight on long-context latency:** keep full attention only at the top and bottom layers (the pattern where empirical gains concentrate), run everything in between as GLA / Gated DeltaNet.

## Adjacent but distinct

- **Sparse attention (MoBA, Longformer):** reduces attention cost by thinning *which tokens* interact, not *which layers* use full attention. Composes with hybrids.
- **MoE:** reduces FFN cost per token, orthogonal to attention cost.
- **KV-cache compression:** reduces memory at inference, not training compute.

## Sources

- Paper: *On-Policy Morphing into Hybrid Attention Models* (FlashMorph2) — Wang, Zheng, Kuang, Tang, Cheng, 2026 (ByteDance Seed · NTU).
- Paper: *Zamba: A Compact 7B SSM Hybrid Model* — Glorioso et al., 2024.
- Paper: *Jamba: A Hybrid Transformer-Mamba Language Model* — Lieber et al., 2024.
- Paper: *The Mamba in the Llama: Distilling and Accelerating Hybrid Models* — Wang et al., 2024 (MOHAWK).

## Conventions

- **Filename:** `_hybrid-attention.md` (underscore prefix for taxonomies).
- **Depth files** for individual recipes land in the same folder; this file keeps the shared mental model.
