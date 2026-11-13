# Layer × module probing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Linear-probe accuracy is often maximized not at the final layer but at a specific (layer, internal module) pair — the output of a particular normalization, attention block, FFN, or residual *inside* a specific block. *Probing Transformer Representations Layer by Layer, Module by Module* (2611.11531) runs over 144K probes across vision foundation models and LLMs and distills two cheap defaults: **final-layer FFN activation for vision models** and **second normalization output at ~60% depth for language models**. Both recover a large share of the grid-search optimum.

**Prereqs:** [README.md](README.md)
**Related:** [../architectures/transformer-block.md](../architectures/transformer-block.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Standard probing picks "the final hidden state" as the probe site. That choice hides a lot: inside each transformer block there are several candidate tensors — the output of each normalization, the attention output, the FFN output, the residual before and after — and the right one depends on both model family and downstream task.

Layer × module probing is simply a grid search over (block index, internal module) pairs, trained with linear probes on the same labels. The gain over "last layer only" is large and systematic.

## How it works

For a given model and downstream task:

1. **Enumerate sites.** For each block, instrument several named taps: `norm1_out`, `attn_out`, `norm2_out`, `ffn_out`, `residual_out`.
2. **Train linear probes.** One probe per (block, module) pair, on the downstream labels, with a fixed held-out split.
3. **Score and pick.** Report the full heatmap; the best site is often not final-layer and often not a residual.

Across the paper's >144K runs, two defaults survive repeatedly:

- Vision foundation models: **FFN output of the final block.**
- Language models: **output of the second layer-norm at ~60% of total depth.**

Both recover a substantial share of the grid-search optimum on new tasks without paying the search cost.

## Why it matters

- Probing is a workhorse of interpretability and transfer-learning diagnostics; picking the wrong site systematically undersells a model's knowledge.
- The defaults are cheap drop-ins with no extra training, and they shift which layers deserve interpretability attention — it is often *not* the final one.
- For vision models, the benefit grows as the gap between pretraining and downstream distribution widens, i.e. exactly where probes matter most.

## Gotchas & tricks

- Modules are not standardized across implementations — some models have pre-norm, some post-norm, some gated-FFN variants. Normalize naming before grid-searching.
- Residuals win on well-matched distributions; FFN/attn outputs win on distribution shift.
- Linear probe only; adding capacity to the probe hides the choice of site.
- For LLMs, the "~60% depth" default is averaged — within a family, re-run the grid once on a small task before committing.

## Sources

- Paper: *Probing Transformer Representations Layer by Layer, Module by Module* — Odonnat, Gnassounou, Chapel, Tavenard, Redko (2026) — [arXiv:2611.11531](https://arxiv.org/abs/2611.11531)
