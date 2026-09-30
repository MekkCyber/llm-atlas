# Adaptive looped transformers (TaH2)
*Depth — per-token adaptive loop depth for test-time compute, trained with lookahead supervision.*

**TL;DR:** Looped transformers reuse the same layer stack for several latent iterations per token to trade decoding FLOPs for accuracy. TaH2 post-trains the backbone jointly with a lightweight **iteration decider** supervised by lookahead labels indicating whether one more iteration would change the prediction. Extra depth is spent only on tokens that benefit.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [prm](prm.md)
**Related:** [orm](orm.md), [mcts](mcts.md)

---

## What it is

A refinement of the looped-transformer family (layer reuse for latent compute) that replaces fixed loop depth with a per-token adaptive schedule. Fixed-depth looping wastes iterations on the majority of tokens whose predictions do not change with more compute; TaH2 spends extra iterations only where they matter.

## How it works

- **Iteration decider.** A tiny head reads the state after each loop iteration and predicts a binary "continue" bit.
- **Lookahead supervision.** During training the model actually runs one extra iteration and compares its prediction to the current one; the resulting label — *did looping change my mind?* — supervises the decider. This is a "did-this-help?" objective, not a value estimate.
- **Joint post-training.** Backbone and decider are optimized together so the backbone learns to make some tokens "loop-friendly" (their predictions shift only with more depth) and others cheap.
- **Inference.** For each token, the model loops until the decider says stop or a max depth is reached, giving a per-token latent compute budget.

## Why it matters

Test-time compute has largely meant "sample more" (best-of-N, self-consistency, tree search) or "think longer" (long-CoT decoding). Adaptive looping opens a third axis — *depth per token* — that is orthogonal to both and doesn't need an external verifier. It also produces a *smoother* accuracy-compute curve than fixed loops, which matters for anytime inference. On AIME, TaH2 improves the accuracy-vs-decoding-FLOP slope from 1.79 to 2.74 (+53%) and beats the non-looped baseline peak by ~3.4 points at matched compute. Gains grow with max depth (from +2.8 at depth 2 to +3.9 at depth 8), whereas fixed-depth loops plateau.

## Gotchas & tricks

- The decider must be lightweight — otherwise its cost eats the FLOP savings. The TaH2 paper uses a single-linear head over the residual state.
- Lookahead labels are noisy at low temperatures (many "same prediction" cases). Curriculum on max depth helps: start at depth 2 so the decider sees clear positive/negative signal, ramp up as the head stabilizes.
- Because the decider was trained with the deployed backbone, changing decoding hyperparameters (temperature, top-p) at inference can shift the decider's calibration. If you swap decoders, retrain the head.
- Interacts with speculative decoding: a draft model can't precompute how many loops the target will run, so any speculation scheme must condition its acceptance rate on the decider's output.

## Sources

- Paper: Improving Test-Time Scaling with Adaptive Looped Transformers — You et al., 2026 — [arXiv:2609.35748](https://arxiv.org/abs/2609.35748)
- Code: [github.com/thu-nics/TaH](https://github.com/thu-nics/TaH)
