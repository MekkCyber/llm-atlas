# Latent Feedback
*Depth — a one-step cross-token recurrence that feeds each token's top-layer hidden state into the next token's input.*

**TL;DR:** A minimal architectural change to a standard Transformer — take the last layer's hidden state at position *t* and inject it (concatenate or add) into the input embedding at position *t+1*. The result is a *latent feedback Transformer* whose forward pass is still parallel over sequence length at training time but gains a shallow recurrence through depth. Across 60M–778M-parameter models it acts as both a **compute amplifier** (reach a given loss with fewer FLOPs) and a **data amplifier** (reach a given loss with less unique data at matched epochs). The authors fit a scaling law that predicts a held-out 778M run's validation loss to within **0.3%**.

**Prereqs:** [transformer-block.md](transformer-block.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md), [../pre-training/README.md](../pre-training/README.md)

---

## What it is

Vanilla Transformers process each position with the same parameters and only communicate *across positions* through attention. Position *t+1*'s input is its token embedding; the model has no mechanism that routes the last layer's answer for position *t* back into the next step's input other than via attention over its own output token. Latent feedback adds exactly that route: the top-layer hidden state `h_t^L` is combined with the token embedding of position *t+1* before the first layer sees it.

This is a form of recurrence, but it preserves training-time parallelism because the top-layer hidden states for all positions are computed in one forward pass and then shifted by one position for the next epoch / inference. It is *not* a full RNN; it is a one-step look-back through depth.

## How it works

Let `e_{t+1}` be the token embedding and `h_t^L` the top-layer hidden state at position *t*. The modified first-layer input is

```
x_{t+1}^0 = f(e_{t+1}, h_t^L)
```

where `f` is a cheap fusion — concat-then-project or a weighted sum with a learned gate. Training is standard next-token prediction. At each step the loss sees the hidden state from the previous step — which during training comes from the previous step in the *same* forward pass (shifted by one position), so no sequential dependency is introduced within a sequence.

At inference the feedback is a natural recurrence: the previous token's `h^L` is already computed and fed into the next step's input.

## Why it matters

The scaling study covers model sizes 60M → 778M and sweeps both unique-token count and epoch count. Two separable effects appear:

- **Compute amplifier.** At matched data and more epochs, latent-feedback models reach a target loss with fewer FLOPs than the baseline.
- **Data amplifier.** At matched compute and fewer unique tokens, latent-feedback models reach a target loss despite the data shortfall, implying better returns on repetition.

Both effects scale with unique-data-per-parameter and with epoch count, and the fitted law predicts held-out 778M results to **0.3%**. One architectural line change that moves both scaling axes at once is a rare and usable result — especially in the current data-constrained regime.

## Gotchas & tricks

- The fusion function `f` matters: concat-then-project preserves more signal than a learned scalar gate, at the cost of a slightly wider first layer.
- Treat `h_t^L` as a *regular* input signal, not a shortcut. Residual connections that bypass it interact unpredictably with the gradient flow at scale.
- Looks like an RNN but isn't: there is no hidden-state update within a position — just one use of the previous position's top state.
- The data-amplifier effect means you can get more mileage out of a fixed corpus; this is most useful on the right side of the Chinchilla curve where unique tokens are scarce.

## Sources

- Paper: *Latent Feedback Is a Compute and Data Amplifier* — Dwivedi, Ma, Moreno, Liu, et al., Institute of Foundation Models (USC) / CMU, 2026 — [arXiv:2611.20404](https://arxiv.org/abs/2611.20404).
