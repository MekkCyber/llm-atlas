# Hybrid linear attention

*Depth — interleaving full-attention and linear-attention / SSM layers in one stack.*

**TL;DR:** Pure linear-attention and SSM (state-space) models scale sub-quadratically but lose recall on long contexts — their fixed-size state can't store the right token to retrieve later. Pure full-attention scales quadratically. *Hybrid* stacks place a small number of full-attention layers among many linear-attention/SSM layers (Jamba, Zamba, HypeNet, Mamba-Transformer hybrids), getting most of the speed of linear attention while the few attention layers handle long-range recall. The hybrid design is the practical sweet spot for cheap long-context as of 2026.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md) · [transformer-block](transformer-block.md)

---

## What it is

Three positions on long-context efficiency:

1. **Pure attention.** Quadratic in sequence length, exact recall. Scaling to 100K+ context is expensive.
2. **Pure linear attention / SSM.** Sub-quadratic, fixed-size state. Tokens written into the state get *summarized* — recall of specific tokens degrades fast.
3. **Hybrid.** Sub-quadratic in most layers + a few full-attention layers that handle the actual long-range routing. Recall lives in the attention layers; speed lives in the linear ones.

The hybrid pattern emerged when pure-linear models hit a recall ceiling and pure-attention models hit a cost ceiling. Hybrids are now the dominant practical answer for $\geq 100$K context.

## How it works

### Layer interleaving pattern

Most hybrids use a ratio like 1:7 (one attention layer per 7 SSM/linear layers) or 1:5:
```
[Linear] × k → [Attention] → [Linear] × k → [Attention] → ...
```

The attention layers are *standard* GQA/MLA; the linear layers are Mamba-2, GLA, RWKV, or similar.

### Routing through Q/K projections

In hybrids, the attention layers' Q and K projections do the heavy lifting for cross-token routing. Long-range retrieval (the model "remembering" a specific earlier token) is implemented as the attention layer's pattern. The cited Attention Amnesia paper (Zhou 2026) pins long-context recall in HypeNet-class hybrids precisely to those Q/K projections.

### Communication between sub-layer types

Linear layers compress information into a fixed state; attention layers can selectively retrieve from any position. The hybrid stack alternates between *compress*-mode and *retrieve*-mode, so the attention layer's KV cache must keep the right tokens accessible.

### State design choices

- **Mamba/SSM state.** Continuous recurrent state of fixed size; updates governed by selective state-space dynamics.
- **GLA / linear-attention state.** A matrix-valued state $S_t = S_{t-1} + k_t v_t^\top$ (or a gated/decayed variant).
- **Hybrid stacking choices.** Whether the SSM layer carries its own positional bias, whether attention layers share KV across heads (GQA/MQA), etc.

## Why it matters

- **Most efficient practical long-context recipe.** At 256K context, hybrids serve at a fraction of full-attention compute with marginal recall loss.
- **Industry adoption.** Jamba (AI21), Zamba, Samba, HypeNet, Mistral's hybrid variants, Kwai's Keye-VL 2.0 long-video stack (sparse attention + MoE hybrid in spirit) all live in this family.
- **The fragile substrate for post-training.** Hybrids' recall lives in a small number of attention layers — making them disproportionately sensitive to fine-tuning that touches those layers. See [qk-restore](../post-training/fine-tuning/qk-restore.md).

## Gotchas & tricks

- **Long-context recall is the failure mode.** Pure-SSM and pure-linear models do well on perplexity but poorly on needle-in-a-haystack (NIAH). Hybrids close the gap; pure-linear doesn't.
- **Ratio matters.** Too few attention layers → recall degrades; too many → speed advantage gone. 1:7 to 1:5 are common sweet spots.
- **Position of the attention layer matters.** Earlier-in-stack attention helps representation building; later helps output routing. Most hybrids place attention layers roughly evenly.
- **CoT-SFT fragility.** Long-CoT supervised fine-tuning biases attention gradients toward short-range patterns and can ablate the long-range routing — the *attention amnesia* phenomenon. Mitigation: [qk-restore](../post-training/fine-tuning/qk-restore.md).
- **Kernel support.** SSM kernels are mature (Mamba-2's Triton kernels are standard); GLA-style kernels are catching up. Hybrid stacks need both kernels well-tuned, not just one.

## Sources

- Paper: *Jamba: A Hybrid Transformer-Mamba Language Model* — AI21 Labs, 2024.
- Paper: *Zamba: A Compact 7B SSM Hybrid Model* — Glorioso et al., 2024.
- Paper: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu & Dao, 2023.
- Paper: *Attention Amnesia in Hybrid LLMs* — Zhou et al., 2026 — [arXiv 2606.11052](https://arxiv.org/abs/2606.11052) — pins recall failure to attention-layer QK projections.
