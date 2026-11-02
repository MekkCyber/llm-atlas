# Soft-MTP

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Vanilla Multi-Token Prediction (MTP) heads optimize cross-entropy on each future token independently. Soft-MTP decomposes that CE gradient into a KL-to-primary-head term (useful self-distillation signal) plus a residual that becomes dominant and noisy at scale, and keeps only the KL part — reconstructed with a straight-through gradient trick that avoids materializing vocab-sized logits. A one-line change, no new parameters, no overhead; reportedly a "free lunch" for anyone training with MTP.

**Prereqs:** [mtp](mtp.md), [transformer-block](../architectures/transformer-block.md)
**Related:** [mtp](mtp.md), [fp8-training](fp8-training.md)

---

## What it is

MTP trains the backbone to predict $k$ tokens ahead through small per-step modules. The paper argues the standard objective — cross-entropy on each future token — is a *mixture* of:

1. a KL from the MTP head toward the primary (next-token) head's distribution (the self-distillation sub-signal), and
2. a residual term that is bounded in theory but becomes noisy and large as the backbone grows, hurting both next-token and multi-token losses.

Soft-MTP strips term (2) and keeps only term (1).

## How it works

Naively, computing the KL would require materializing both heads' full vocabulary-sized distributions at every position — prohibitive at frontier vocab sizes. The paper's "gradient-recovery" trick notes that the KL gradient can be reconstructed from the already-computed CE gradients of each head via a straight-through routing: cross-entropy forward, KL gradient on the backward. Pseudocode:

```
# per MTP position
logits_main = main_head(h_main).detach()         # teacher
logits_mtp  = mtp_head(h_mtp)
loss_mtp    = ce(logits_mtp, target) \
            - ce(logits_main, target).detach() \   # removes the noisy residual
            # with straight-through so the backward is KL(p_main || p_mtp)
```

Zero parameter overhead, no new buffers, same compute as vanilla MTP.

## Why it matters

- MTP is now a frontier default (DeepSeek-V3) because of the free self-speculative-decoding draft head. Soft-MTP keeps all of that and adds a cleaner training signal.
- Reframes MTP as a *self-distillation* technique rather than a parallel-prediction trick, which clarifies why it helps backbone representations: it is the model teaching itself from its own best head.
- Zero deployment cost — the serving stack doesn't change, only the training loss.

## Gotchas & tricks

- The gradient-recovery trick depends on the primary head and MTP heads sharing the output projection; if you un-tie them you lose the equivalence and the trick becomes an approximation.
- Reported gains scale with model size: the residual term is small at ≤200M but increasingly dominant past ~1B. Expect bigger lifts at frontier scale, not at pilot runs.
- Only touches the MTP positions; the main next-token CE at position $t$ is unchanged.
- Compatible with DeepSeek-V3's late-training $\lambda$ decay; the paper does not comment on changing the schedule.

## Sources

- Paper: *Multi-Token Prediction Done Right* — Teoh et al., MIT, 2026 — [arXiv:2610.38331](https://arxiv.org/abs/2610.38331).
- Prior: *Better & Faster LLMs via Multi-token Prediction* — Gloeckle et al., Meta, 2024 — the baseline MTP objective.
- Prior: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — depth-1 MTP in production.
