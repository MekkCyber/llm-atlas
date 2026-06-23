# Reflective Masking

*Depth — a multi-turn re-mask-and-denoise post-training recipe that gives mask diffusion models a native test-time-scaling story for reasoning.*

**TL;DR:** Autoregressive LMs scale reasoning at test time by emitting long chains of thought. Mask diffusion models (MDMs) have no equivalent — until **Reflective Masking (RM)**: a lightweight post-training recipe that teaches an MDM to *revisit and locally edit its own prior outputs* across multiple denoising rounds. Each round re-masks a subset of low-confidence or constraint-violating tokens and denoises again, conditioned on the previous draft. A parameter-free *History Reference* mechanism reuses intermediate denoising states as cross-turn memory. Demonstrated across text, Sudoku, and image editing (Zhang et al., 2026).

**Prereqs:** [../../fundamentals/mask-diffusion-lm.md](../../fundamentals/mask-diffusion-lm.md)
**Related:** [long-cot-rl](long-cot-rl.md), [length-penalty](length-penalty.md)

---

## What it is

A multi-turn inference loop on top of an MDM:

```
Turn 0: denoise from all-mask → draft x₀
Turn 1: re-mask a subset S₁ of x₀'s tokens → denoise → draft x₁
Turn 2: re-mask a subset S₂ of x₁ → denoise → draft x₂
...
Turn k: until convergence or budget
```

Two design choices define a specific RM variant:

- **The re-masking policy.** Which tokens to re-mask on each turn? Low model confidence is the default; constraint violations (e.g. a Sudoku row sum) are an alternative; learned classifiers are a third.
- **The cross-turn memory.** Without explicit memory, each turn starts from scratch given the previous draft. *History Reference* exposes intermediate denoising states from prior turns to the current denoising step, parameter-free.

A lightweight SFT post-training stage teaches the model how to use the re-masking signal — without it, the model treats each turn independently and gains little from iteration.

## How it works

Per-turn loop:

```
input:  current sequence x_t  (some tokens committed, some `[MASK]`)
        history H_t (intermediate states from turns 0..t-1)

1. compute denoising distribution p_θ(· | x_t, H_t)
2. apply re-masking policy:
       - score each currently-committed token (confidence / constraint / classifier)
       - mask the lowest-scoring s%
3. sample / commit predictions for masked positions → x_{t+1}
4. push x_t into H_{t+1}
5. terminate if Δ between turns < ε or budget exhausted
```

History Reference is the parameter-free attention mechanism that lets the current denoising step attend to the cached representations from prior turns — analogous to how AR reasoning re-uses earlier CoT tokens as context.

## Why it matters

RM unlocks a clean *test-time scaling* knob for MDMs:

- **More turns → better answers** on tasks where corrections are local (a few wrong tokens in an otherwise good draft). This is the regime where AR reflection is wasteful (regenerate the entire CoT) but RM is efficient (only re-touch broken spans).
- **No architectural changes.** Lightweight post-training is enough; existing MDM backbones can adopt RM.
- **Generality across modalities.** Demonstrated on text, Sudoku (combinatorial), and image editing — the same primitive applies.

If RM holds up at scale, it gives MDMs a credible answer to "where's your CoT story?" — and that answer is differently shaped (iterative refinement instead of token-sequential thinking), with a plausible computational advantage on tasks where most of the work is *local fixing* rather than long-range planning.

## Gotchas & tricks

- **Re-masking policy is the dominant hyperparameter.** Too aggressive (mask everything) is wasteful; too conservative (mask only the worst 1%) converges to no-op. The paper finds task-dependent sweet spots.
- **History Reference has compute cost.** Stacking too many turns of history blows up the effective context window. Practical implementations cap history depth.
- **Confidence ≠ correctness.** Low-confidence tokens *can* be right; high-confidence tokens *can* be wrong. The re-masking policy can be improved with a separate verifier signal.
- **Pairs with constraint signals when available.** For tasks with hard constraints (Sudoku, code syntax, JSON), use violation as the re-masking signal rather than confidence — much stronger.

## Sources

- Paper: *Multi-Turn Reflective Masking Elicits Reasoning in Mask Diffusion Models* — Zhang, Bian, Qi, Yao, Huang, Zhou, 2026 — https://arxiv.org/abs/2606.16700
