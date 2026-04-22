# Z-Loss
*Depth — auxiliary loss that keeps the log-partition function from drifting.*

**TL;DR:** Add `α · (log Z)²` to the training loss, where `Z = Σ exp(logits)` is the softmax partition function over the vocabulary. Keeps the logit vector's overall magnitude from drifting upward during long training runs, preventing the softmax from becoming pathologically sharp and killing gradients. Used by PaLM, OLMo 2, and most stability-conscious large runs.

**Prereqs:** [attention](attention.md), [tokenization](_tokenization.md)
**Related:** [qk-norm](../architectures/qk-norm.md), [reordered-norm](../architectures/reordered-norm.md), [_training-stability](../pre-training/_training-stability.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

The cross-entropy loss for next-token prediction is

```
L_ce = -log softmax(logits)[target]
     = -logits[target] + log Z
     with  Z = Σ_v exp(logits[v])
```

`log Z` is the log-partition function — a scalar summary of "how large the logits are overall". Cross-entropy cares only about the *difference* `logits[target] − log Z`, so it doesn't pin down the absolute magnitude of the logits. Nothing stops the logits (and `log Z` with them) from drifting upward indefinitely.

Z-loss adds a soft constraint that `log Z` stays near zero:

```
L = L_ce + α · (log Z)²
```

Typical `α ≈ 1e-4`. Applied only to the final output logits, not intermediate projections.

## How it works

### Why drift happens

Cross-entropy has a **gauge freedom**: adding a constant `c` to all logits doesn't change the softmax, so doesn't change CE loss. But it changes `log Z` by exactly `c`. During training, the model has no incentive to keep logits centered — AdamW's weight decay on the LM head pulls toward zero, but gradient signals can still push them up, especially when the model is learning to be confident on common tokens.

### What drift breaks

Two failure modes, both from the softmax saturating:

1. **Gradient collapse.** When `exp(logit_max) ≫ Σ_{v≠max} exp(logit_v)`, the softmax output is (≈1, 0, ..., 0). The gradient of `softmax(z)_i` with respect to `z_j` is `softmax(z)_i (δ_ij − softmax(z)_j)`. At saturation this is ≈0 everywhere except the argmax entry. The LM head stops receiving useful signal.
2. **Numerical instability.** `log Z` drifting into the hundreds means `exp(logit_max)` overflows in fp16/bf16. Mixed-precision training blows up with NaN.

### What z-loss does

The gradient of `α · (log Z)²` with respect to the logits is

```
∂/∂logits[v] α(log Z)²  =  2α · log Z · softmax(logits)[v]
```

In words: if `log Z > 0` (logits too big), push every logit down by an amount proportional to its probability. If `log Z < 0`, push up. Result: `log Z` wiggles around zero, logits stay in a bounded range, softmax never saturates.

The constraint is soft — z-loss allows `log Z` to move by O(1) but penalizes growth into the tens or hundreds. That flexibility matters: pinning `log Z` exactly to zero would be too restrictive.

### Implementation (PyTorch)

```python
import torch
import torch.nn.functional as F

def loss(logits, targets, z_loss_weight=1e-4):
    # logits:  [B, T, V]  float
    # targets: [B, T]     LongTensor of token ids (NOT one-hot)
    log_z = torch.logsumexp(logits, dim=-1)                 # [B, T], numerically stable
    log_probs = logits - log_z.unsqueeze(-1)                # [B, T, V]
    ce = F.nll_loss(log_probs.flatten(0, -2),               # [B*T, V]
                    targets.flatten(),                      # [B*T]   gathers by index
                    reduction="none").view_as(targets)      # [B, T]
    z = z_loss_weight * log_z.pow(2)                        # [B, T]
    return (ce + z).mean()
```

## Why it matters

- **Removes a common source of mid-run NaN.** Many "our training run died at step 400k" post-mortems trace back to drifting logit magnitudes.
- **Complements [QK-norm](../architectures/qk-norm.md).** QK-norm bounds *attention* logit magnitudes; z-loss bounds *output* logit magnitudes. Two different layers, same family of failure.
- **Almost free.** One extra reduction and a scalar square per batch.
- **No hyperparameter pain.** `α = 1e-4` works across model sizes. Not a tuning knob.

## Gotchas & tricks

- **Apply only to the final logits.** Don't add z-loss to intermediate projections or logits of auxiliary heads unless those heads have the same saturation risk.
- **Use logsumexp.** Computing `log Σ exp(logits)` directly overflows; `logsumexp` is numerically stable.
- **Reduction order.** Compute z-loss per sequence, then average — same reduction as CE. Don't sum z-loss across the batch and CE across the batch but with different weights.
- **Interaction with weight tying.** If LM head is tied to the input embeddings, z-loss still works (it affects the logits via the shared matrix).
- **Don't confuse with "router z-loss" in MoE.** MoE literature uses a similarly named term for the router's load-balancing logits. The idea is the same (bound `log Z`) but applied to the routing distribution, not the LM head.

## Sources

- Paper: *PaLM: Scaling Language Modeling with Pathways* — Chowdhery et al., 2022 — introduces z-loss into pre-training at scale.
- Paper: *2 OLMo 2 Furious* — AI2, 2024 — documents z-loss as part of the stability package.
- Paper: *ST-MoE* — Zoph et al., 2022 — router z-loss variant for MoE load balancing.
