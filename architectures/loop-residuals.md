# Loop Residuals (LoopRes)
*Depth — attention residuals that let each looped layer attend directly to earlier loops' outputs with structural tags.*

**TL;DR:** Looped Transformers iterate a shared block $L$ times to add compute without parameters. Prior work needed to train from scratch and barely helped pretrained LLMs. LoopRes (2026) makes looping effective at **finetune** time by adding Attention Residuals that let each looped layer attend to all earlier loops' outputs, each tagged with a learnable loop-aware key, within a sliding residual window. Across 6 LLMs and 9 benchmarks, LoopRes at 2× compute improves Qwen3-1.7B-Base by 29.1% over SFT, more than doubling the previous SOTA looped-finetune baseline.

**Prereqs:** [transformer-block](transformer-block.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [../pre-training/mtp](../pre-training/mtp.md)

---

## What it is

An architectural modification applied when converting a pretrained LLM into a depth-recurrent / looped Transformer. The base block is iterated $L$ times. On each iteration, the attention sub-layer can read not only the current loop's tokens but also the outputs of **earlier loops**, each labeled with a learnable key encoding its loop index. A residual window restricts attention to the most recent $W$ loops so that each loop builds incrementally on recent context.

## How it works

Augmentations per looped layer:

1. **Loop-aware keys.** For each past loop $\ell$, learn a key vector $k_\ell$ added to the attention keys of iteration $\ell$'s outputs. This disambiguates "which loop produced this state" so the attention can treat iterations as distinguishable timesteps rather than indistinguishable iterates.
2. **Attention residuals.** The query at loop $\ell$ can attend to keys/values from loops $\ell - 1, \ell - 2, \ldots, \ell - W$, not only the current loop. This is the "residual" pathway through loop-space.
3. **Residual window $W$.** Capped to force each loop to depend on recent iterations, not skip to earlier ones. Prevents degenerate solutions where the model just copies through.

Initialization: all added parameters start near zero so the finetune starts from the pretrained behavior and monotonically adds looping capacity.

## Why it matters

Looped Transformers were a strong "free compute" idea blocked by two facts: training from scratch is expensive, and naive looped-finetuning gave marginal gains. LoopRes removes both blockers. Reported scaling behavior:

- Qwen3-1.7B-Base: +29.1% over SFT at 2× compute.
- Gains hold across Qwen3 (up to 30B), and two other model families.
- From 1.7B to 8B, LoopRes scales *more steeply* with compute than SFT — a higher ceiling with less compute on long outputs.

This is a test-time-scaling axis that layers on top of pretraining with cheap finetuning, so it composes with every other post-training recipe.

## Gotchas & tricks

- **Loop-aware keys are essential.** Without them, the model cannot tell which iteration produced a state and the residual pathway collapses.
- **Window $W$ is a bias-variance knob.** Too wide and the model skips iterations; too narrow and the residual pathway is useless. The paper sweeps $W$ per model family.
- **Compute trade.** Each loop is a full forward pass; $L$ loops cost roughly $L\times$ the base model's inference compute. Gains vs. compute cost tilt in LoopRes's favor on long outputs where latency matters less.
- **Pipeline parallelism** for these models is non-trivial — standard 1F1B schedules stall. See [systems/loop-pp](../systems/loop-pp.md).

## Sources

- Paper: Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals — Zhao et al., 2026 — [arXiv:2611.31094](https://arxiv.org/abs/2611.31094)
