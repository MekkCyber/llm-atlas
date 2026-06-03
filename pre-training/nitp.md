# NITP — Next Implicit Token Prediction
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pretraining-objective augmentation. Standard next-token prediction (NTP) supervises only through discrete labels in logit space, leaving hidden representations under-constrained and prone to anisotropic / degenerate geometry. NITP adds a *dense, continuous* auxiliary loss in **representation space**: predict the implicit semantic content of the next token using **shallow-layer hidden states of the same model** as stable self-supervised targets. ~2% extra training FLOPs, zero inference overhead, +5.7 abs on MMLU-Pro for a 9B MoE.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [README](README.md)
**Related:** [mtp](mtp.md), [../fundamentals/z-loss.md](../fundamentals/z-loss.md)

---

## What it is

NTP's cross-entropy loss is sparse: at each position, only one logit's value matters for the gradient (the true next-token's). Hidden states have far more degrees of freedom than this one-dimensional signal pins down, leaving the latent geometry under-constrained — empirically resulting in anisotropic ("token embeddings cluster in a narrow cone") and degenerate representations that limit downstream transfer. NITP augments the loss with a dense supervision signal *in representation space*: at each position the model must additionally predict the *next token's representation*, where the target is the same model's own shallow-layer output for that token. This is a cheap, drop-in addition — no extra targets, no teacher model, no inference overhead.

---

## How it works

### The auxiliary loss

For token $x_{t+1}$ at position $t$:

- **Target:** $z_{t+1} = h_\ell(x_{t+1})$ — the model's own *shallow-layer* hidden state (e.g. layer $\ell = 2$ or 4) for $x_{t+1}$ in its full context. The shallow layer is stable across training (it's close to the input embedding) and not training-dynamic.
- **Prediction:** $\hat z_{t+1} = f(h_L(x_{\le t}))$ — a small head on top of the final-layer hidden state at position $t$.
- **Loss:** a continuous distance — usually cosine + MSE — between $\hat z_{t+1}$ and $z_{t+1}$, with the target detached (`stop_grad`).

Total: $\mathcal{L} = \mathcal{L}_{NTP} + \lambda \cdot \mathcal{L}_{NITP}$.

### Why shallow targets

Shallow layers are close to the input embedding, so $z_{t+1}$ is a stable, predictable representation that early in training already encodes lexical / morphological structure and later acquires syntactic / semantic content. Using a *deep* layer as the target would create a chicken-and-egg situation (target shifts as the model learns). Using the input embedding directly is too easy (the model already has access to embeddings).

### Theoretical claim

The paper argues NITP regularizes the optimization landscape: NTP leaves directions in latent space along which loss is flat (under-constrained), which the model wanders into degenerate / anisotropic geometry. Dense regression in representation space pins those directions and enforces a compact, structured geometry.

### Cost

- **Training:** ~2% extra FLOPs — one small head, one shallow-layer forward pass on already-computed inputs.
- **Inference:** zero — the head is discarded.

---

## Why it matters

- **Tiny modification, real downstream gains.** Across dense and MoE models from 0.5B to 9B, consistent improvement. On a 9B MoE: **+5.7 abs MMLU-Pro, +6.4 C3, +4.3 CommonsenseQA**, with no inference cost.
- **Joins a small family of cheap pretraining regularizers.** Conceptually adjacent to [z-loss](../fundamentals/z-loss.md) (logit-norm regularizer) and to dense pretraining auxiliaries in vision (DINO-style self-distillation). NITP is the LLM equivalent of "regress against your own shallow features."
- **Orthogonal to [MTP](mtp.md).** MTP densifies supervision in *token* space (predict tokens $t+2, t+3, \ldots$); NITP densifies in *representation* space (predict the representation of $t+1$). They could compose.
- **Fixes a known geometry problem.** Anisotropy of transformer embeddings has been documented since Gao et al. 2019 (BERT) and Ethayarajh 2019 (GPT-2). NITP is one of the few interventions that addresses it during pretraining rather than post-hoc.

---

## Gotchas & tricks

- **Detach the target.** $z_{t+1}$ is a target, not a prediction — must use `stop_grad`, or the shallow layer learns to produce easy targets for the head, defeating the purpose.
- **Pick the shallow-layer index empirically.** Too shallow (layer 0–1) and the target is trivially close to embeddings; too deep and the target moves during training. Layers 2–4 typical for ~20-40 layer models.
- **$\lambda$ tuning.** Too small and the regularizer doesn't bite; too large and it competes with the main task. Modest values (~0.1–1.0) work in the paper.
- **MoE models benefit more than dense.** The 9B MoE gets the biggest jump; the authors hypothesize that MoE's sparser activation patterns make the latent geometry even more degenerate without NITP.
- **Not the same as MTP.** MTP predicts future tokens via auxiliary modules; NITP predicts the *representation* of one future token via a small head. Different mechanism, different cost profile (NITP is cheaper).
- **Inference overhead is exactly zero** — the auxiliary head and the shallow-layer reads are training-only.

---

## Sources

- Paper: *NITP: Next Implicit Token Prediction for LLM Pre-training* — multi-institution, 2026 — [arXiv:2605.24956](https://arxiv.org/abs/2605.24956).
- Code: https://github.com/aHapBean/NITP
- Background: *How Contextual are Contextualized Word Representations?* — Ethayarajh, EMNLP 2019 — documents transformer-embedding anisotropy that NITP regularizes against.
- Adjacent: *Multi-Token Prediction* — Gloeckle et al., 2024, see [mtp](mtp.md) — orthogonal densification of the pretraining signal.
