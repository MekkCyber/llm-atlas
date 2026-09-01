# Ring Forcing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A long-horizon autoregressive video diffusion recipe that attacks two failure modes at once — **object permanence** (do things look the same when they come back on screen?) and **memory capacity** (can the model even see far enough back?). Combines a **ring-structured training curriculum** that forces retrieval from distant history, a **compression + timestep-composition** scheme that stretches the effective history to minutes under fixed sequence length, and a **sparse RoPE** variant for flexible memory adaptation without breaking the pretrained backbone's positional priors.

**Prereqs:** [README.md](README.md), [../fundamentals/rope.md](../fundamentals/rope.md), [../fundamentals/_positional-encoding.md](../fundamentals/_positional-encoding.md)
**Related:** [layerrecall.md](layerrecall.md)

---

## What it is

Long autoregressive video diffusion breaks in two distinguishable ways: identity drift when subjects re-enter (object permanence) and outright inability to see the relevant history at all (memory capacity). Recipes that only fix one collapse on the other — perfect permanence with no capacity limits temporal scope, huge capacity without permanence smears identity. Ring Forcing addresses both.

## How it works

Three integrated ingredients:

1. **Ring-structured training strategy.** The training curriculum arranges historical chunks in a ring and forces the model to retrieve from *distant* segments, not just the recent recency-cached ones. This resolves the trade-off between strict historical adherence (over-copying) and generative diversity (drift) by making distant retrieval a routine training signal.
2. **Compression + timestep composition.** Under fixed sequence-length constraints, compress older history and *compose* it across denoising timesteps so the effective historical span reaches **minutes-long durations**. Each attention step sees a global receptive field over the entire history at a cost that stays within the model's budget.
3. **Sparse RoPE.** A sparse variant of Rotary Position Embedding that gives memory tokens flexible, scalable position indices without disturbing the pretrained backbone's positional priors — so you can add memory to an existing model instead of retraining from scratch.

## Why it matters

- State-of-the-art **minutes-long coherence** and object permanence on the paper's evaluations, significantly outperforming prior long-video AR diffusion baselines.
- The ring curriculum is a **training-side** fix — pairs with architectural fixes like LayerRecall's layer-selective router rather than competing with them.
- Sparse RoPE preserves pretrained priors, which is what makes the recipe applicable to existing video foundation models rather than only to models trained from scratch.

## Gotchas & tricks

- Ring curriculum design (chunk size, ring radius, sampling frequency of distant retrievals) is what determines how well permanence transfers to unseen clip lengths.
- Compression is lossy — the composition-across-timesteps trick partly compensates by amortizing the loss across the denoising trajectory. Aggressive compression alone regresses on object identity.
- Sparse RoPE's position indices for memory tokens must be chosen so the model doesn't see relative offsets far outside its training regime; the paper's protocol matters.

## Sources

- Paper: *Ring Forcing: Towards Precise Long-Term Memory for Autoregressive Video Diffusion* — Xue et al., Stanford / MIT / Peking / UCB / ByteDance, 2026 — [arxiv](https://arxiv.org/abs/2608.26794)
