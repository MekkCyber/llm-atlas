# FLaRe — Flow-based Latent Reasoning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** FLaRe is a latent-reasoning recipe that *thinks* in a learned continuous space via **flow matching** and verbalizes only the final answer. It enumerates five requirements a latent thought must meet (**useful, diverse, explainable, refinable, efficient**) and makes training choices that satisfy each — including a final stage of self-training on the model's own verified thoughts. At inference it reaches **97% of explicit-CoT accuracy at ~25% of its latency**.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [../_post-training.md](../_post-training.md)
**Related:** [mcts.md](mcts.md) · [length-penalty.md](length-penalty.md) · [prm.md](prm.md)

---

## What it is

Explicit chain-of-thought (CoT) spends many decoded tokens on reasoning before the answer. **Latent reasoning** runs the "thought" through continuous activations instead and decodes just the answer, cutting latency. Prior latent methods tend to collapse into shortcuts (distilling an explicit CoT into weights, or reusing the question's embedding).

FLaRe argues that a useful latent reasoning method must simultaneously satisfy five explicit properties:

1. **Useful** — the latent genuinely drives correct answers, not just changes them.
2. **Diverse** — resampling yields different reasoning trajectories (not a degenerate shortcut).
3. **Explainable** — a decoded CoT reflects what the answer actually follows.
4. **Refinable** — more inference compute keeps improving accuracy.
5. **Efficient** — costs less than an equivalent-accuracy explicit CoT.

FLaRe is a recipe that targets all five via flow matching in a learned latent space.

## How it works

- **Latent space.** The latent $z$ is learned (not fixed to an embedding of the question), so the network can allocate it to reasoning content rather than redundant input tokens.
- **Flow training.** A flow matching objective is trained in that latent space — $z$ is the sample path from a simple prior to a reasoning-conditional target. Flow matching gives a stable continuous-time surrogate for diffusion-style training without discrete denoising steps.
- **Readout.** After flowing, a decoder reads off the answer from $z$. The decoded CoT at readout time can be inspected (explainability probe).
- **Self-training stage.** A final training phase trains on **the model's own thoughts, filtered by verifier**. This is the recipe's analog of RLVR for latent reasoning and is what makes the latent *useful* (not just plausible).

## Why it matters

- Closes most of the quality gap latent reasoning has had with explicit CoT: 97% accuracy at ~25% latency on arithmetic benchmarks.
- Makes latent reasoning **probeable**: the "explainable" requirement is now operationalized by decoding what the latent corresponds to, addressing the main critique of continuous CoT.
- Flow matching avoids both one-token-at-a-time imitation (which hurts diversity) and shortcut distillation (which hurts usefulness).

## Gotchas & tricks

- **Verifier matters for the self-training stage.** Without a verified-answer filter, the self-training step risks collapsing diversity or converging to easy-but-wrong modes — a continuous analog of the TTT self-feedback failure (see [../test-time-training.md](../test-time-training.md)).
- **Latent dim is a quality knob.** Too small → runs out of capacity for reasoning. Too large → explainability decoding gets noisy.
- **Not a free CoT replacement.** On very hard problems where explicit CoT scales linearly with difficulty, FLaRe plateaus faster — reserve for latency-sensitive settings.
- **Composes with RL.** FLaRe recipe + GRPO on the decoder side is a natural stack; the paper's self-training stage is already in that spirit.

## Sources

- Paper: *What Matters for Latent Reasoning with Flow Matching* — 2026 — [arXiv:2610.06666](https://arxiv.org/abs/2610.06666).
