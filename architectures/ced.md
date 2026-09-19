# Causal Encoder-Decoder (CED)
*Depth — asymmetric prefill/decode compute for agent-shaped LLM workloads.*

**TL;DR:** Standard decoder-only LLMs use the same weights (and thus the same activated FLOPs per token) for prefill and decode. Agent workloads have input-heavy prompts that make prefill the dominant cost. **CED** splits the model into a lighter *encoder-like* prefill tower and a heavier *decoder-like* decode tower, so prefill activates fewer parameters per token than decode. In DeepSeek-V4.1-Flash (2026): 8B active during prefill vs. 16B during decode, over a 552B backbone. The two towers share the same KV cache path (CSA2), so the split is transparent to the attention layer.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [_moe](_moe.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [csa2](csa2.md), [deepseek-moe](deepseek-moe.md), [../case-studies/deepseek-v4-1-flash.md](../case-studies/deepseek-v4-1-flash.md)

---

## What it is

An architectural split of a decoder-only transformer into two asymmetric compute pathways:

- **Prefill tower.** Runs on the full prompt in one pass. Activates a *smaller* subset of MoE experts per token. Cheaper.
- **Decode tower.** Runs one token at a time during generation. Activates a *larger* subset. More expressive per token.

Both towers share the same tokenization, embedding, and — crucially — the same KV cache produced by the attention layers. It's not an encoder-decoder in the T5 sense (there is no cross-attention); it's a causal decoder-only model with two activation regimes.

The name "Causal Encoder-Decoder" reflects the roles (prefill *encodes* the input state; decode *generates* from it) while preserving the causal-decoder architecture.

---

## How it works

At each transformer layer:

```
input token ─► attention (shared KV cache via CSA2)
                 │
                 ▼
              MoE gate
                 │
        ┌────────┴────────┐
        │                 │
   prefill regime    decode regime
   activate expert   activate expert
   subset E_prefill  subset E_decode
   (fewer, cheaper)  (more, richer)
```

The gate reads a *phase flag* (are we in prefill or decode?) and routes to the appropriate expert subset. The two subsets can overlap or be disjoint; V4.1-Flash's exact partition is not enumerated in the abstract.

At sequence-parallelism time (prefill runs the full prompt in one shot), all tokens use the prefill subset simultaneously. At autoregressive decode, one token at a time uses the decode subset.

Because the attention layer is shared and the KV cache is shared, switching regimes has zero data-movement cost — it's a routing decision, not a model swap.

---

## Why it matters

- **Directly attacks prefill cost.** For an agent that reads a 500k-token codebase then emits 500 tokens, prefill FLOPs dwarf decode FLOPs. CED cuts them in half at the activation level.
- **Preserves decode quality.** Decode is where generation quality is expressed; CED gives decode the full expert budget while shrinking prefill's.
- **Composes with CSA2.** CSA2 shrinks the KV cache; CED shrinks activated FLOPs. Both target agent workloads, from different sides.
- **Serves as a template.** Any MoE model where prompts are input-heavy could adopt the same asymmetric-gate trick. Expect follow-ups from other MoE lines.

---

## Gotchas & tricks

- **Prefill/decode partition is a design decision.** Too aggressive a prefill trim hurts input understanding; too little defeats the point. V4.1-Flash's 8B/16B split is one point on a Pareto curve.
- **Training must see both regimes.** Naively training only in decode mode leaves prefill experts under-trained. Expect scheduled prefill passes during pretraining, or dedicated prefill-mode SFT.
- **Serving stack must handle regime switches.** Batching a mix of prefill and decode requests requires the runtime to route to different expert subsets simultaneously. Custom kernels likely required.
- **KV cache sharing is the invariant.** If the two towers diverge in attention shape, the CSA2 sharing breaks. Keep attention paths identical.
- **Cost split ≠ quality split.** The prefill tower is asked to *represent*, not to generate; the quality signal for the smaller subset is representation quality, not perplexity. Evaluate accordingly.

---

## Sources

- Paper: *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* — DeepSeek-AI, 2026 — [arXiv:2609.19969](https://arxiv.org/abs/2609.19969).
- Case study: [deepseek-v4-1-flash](../case-studies/deepseek-v4-1-flash.md).
