# QK-Restore

*Depth — training-free fix for long-context recall loss caused by CoT SFT in hybrid LLMs.*

**TL;DR:** CoT supervised fine-tuning of [hybrid linear-attention](../../architectures/hybrid-linear-attention.md) models catastrophically degrades long-context recall — HypeNet-9B drops from 67.2% to 9.4% on NIAH-S2@256K after CoT-SFT. The damage localizes to the attention layers' Q and K projections. QK-Restore copies only the pre-SFT Q and K weights back over the post-SFT model, leaving the rest of the SFT-trained weights intact. Reasoning ability preserved, long-context recall recovered, zero training cost.

**Prereqs:** [../../architectures/hybrid-linear-attention](../../architectures/hybrid-linear-attention.md)
**Related:** [README](README.md)

---

## What it is

Hybrid linear-attention models route long-range retrieval through the few full-attention layers in the stack. The attention layers' Q and K projections decide *which* earlier tokens get pulled back. Long-CoT SFT biases attention gradients toward short-range patterns (every token in a 4000-token chain attends mostly to the previous few tokens), which overwrites these long-range routing patterns. The result: the model still reasons well in-window, but loses the ability to retrieve from far back.

QK-Restore is a two-line patch: keep all SFT-trained weights *except* the attention layers' $W_Q$ and $W_V$ matrices, which are restored from the pre-SFT checkpoint. The pre-SFT routing pattern is back; the SFT-acquired reasoning is unaffected.

## How it works

### Diagnosis

The paper localizes the failure by ablating SFT-induced weight changes layer-by-layer and projection-by-projection. The drop in NIAH score cleanly attributes to attention layers' Q and K. V and the FFN are unaffected.

### Procedure

Given pre-SFT checkpoint $\theta_\text{pre}$ and post-SFT checkpoint $\theta_\text{post}$:

```python
theta_restored = deepcopy(theta_post)
for layer in attention_layers(model):
    theta_restored[layer.W_q] = theta_pre[layer.W_q]
    theta_restored[layer.W_k] = theta_pre[layer.W_k]
# All other weights — V, output projection, SSM/linear layers, FFN — stay from post-SFT
```

Apply the restored model at inference. No retraining, no calibration.

### Why only Q and K

- **V** carries token content, not routing — SFT modifications to V usually reflect the new vocabulary/style and shouldn't be reverted.
- **FFN and SSM** layers carry the reasoning capability the SFT installed.
- **Attention output projection** is downstream of routing; reverting it would interact with the rest of the layer ambiguously.

Reverting Q and K alone restores the pre-SFT *routing* pattern while preserving the SFT-trained *content* and *reasoning* pipeline.

## Why it matters

- **Recovers a real, large performance gap at zero cost.** HypeNet-5B S3@256K: 65.4 → 76.4. HypeNet-9B from collapsed to near pre-SFT.
- **Establishes a precise mechanism for a known phenomenon.** Hybrid LLMs' "fine-tuning regression" was widely observed; QK-Restore is the first clean attribution + fix.
- **Likely generalizes.** The localization argument applies to any hybrid stack where attention layers carry the long-range routing — Jamba, Zamba, Samba, etc. — though specific layer choices may vary.

## Gotchas & tricks

- **Hybrid-specific.** The fix is for *hybrid* stacks where a few attention layers carry recall. Doesn't apply to pure-attention models (where SFT effects spread across all layers).
- **Don't restore V.** Reverting V hurts the model's reasoning-quality score — V carries content the SFT properly updated.
- **Reasoning benchmarks unaffected.** GSM8K, MATH, code benchmarks all stay at post-SFT levels. The intervention is precisely targeted at long-context routing.
- **Verify on your own checkpoint.** The clean Q/K-only attribution was demonstrated on HypeNet. Other hybrid families may have slightly different damaged-projection sets — ablate before deploying.
- **Doesn't prevent the damage.** It only undoes it. A QK-aware SFT (regularizing the attention layers' Q/K toward pre-SFT) would prevent the regression in the first place. Open work.

## Sources

- Paper: *Attention Amnesia in Hybrid LLMs: When CoT Fine-Tuning Breaks Long-Range Recall, and How to Fix It* — Zhou et al., HKUST(GZ) / Mistral AI et al., 2026 — [arXiv 2606.11052](https://arxiv.org/abs/2606.11052).
