# Looped Transformers
*Depth — recurrent-block transformer architectures that trade depth for shared parameters executed many times.*

**TL;DR:** A **looped transformer** executes a single shared transformer block $B$ for $N$ iterations, so an $L$-effective-layer model uses only $L/N$ worth of parameters. The recipe shows up in **Ouro** and **Huginn**, among others, and competes with standard dense transformers at reasoning tasks that reward iterative refinement. Each loop produces a decodable next-token distribution, which recent decoding work (LoopCD) exploits.

**Prereqs:** [transformer-block.md](./transformer-block.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [../inference/contrastive-decoding.md](../inference/contrastive-decoding.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A dense transformer stacks $L$ *distinct* blocks. A looped transformer stacks $L/N$ distinct blocks and runs each $N$ times, with hidden-state carry between iterations. Shared variants in the literature:

- **Universal Transformer** (Dehghani 2019) — one block looped a dynamic number of times, with per-position halting.
- **Albert / Ouro / Huginn** — one block looped a fixed number of times, often with adaptive compute heads.
- **"Looped" scaling studies** that interpolate between depth and loop count.

The parameter savings are large (nothing new to learn per extra loop), and the shared block tends to specialize into an "iterative refinement" circuit.

## How it works

For input $x$ and shared block $B$:

$$
h_0 = \text{embed}(x), \qquad h_{i+1} = B(h_i) \;\; \text{for } i = 0, \ldots, N-1, \qquad y = \text{lm\_head}(h_N)
$$

Key design knobs:

- **Static vs adaptive $N$.** Static — fixed loops per example (simple). Adaptive — a halting head decides per token or per sequence when to stop.
- **Residual persistence.** Some variants concatenate or project $h_0$ into every loop ("input feeding") to prevent drift.
- **Decodable intermediate loops.** Each $h_i$ can be sent through the LM head to produce a logit distribution, which is the basis for LoopCD.
- **Training.** Typically trained with the final-loop loss, sometimes with auxiliary losses at intermediate loops to make them individually decodable.

## Why it matters

- **Parameter efficiency.** A looped-$N$=8 model matches a dense-$L$=$8k$ model's reasoning benchmark on far fewer unique parameters, trading compute for storage — attractive for edge deployment and for models that need to fit in a tight memory envelope.
- **Iterative-refinement bias.** Looped models seem to improve reasoning more smoothly with more compute, mimicking how CoT improves with more tokens.
- **Latent weak-and-strong pairs.** Earlier loops are predictions of the same next token with less computation — a built-in "weaker expert" that enables contrastive decoding without a second model.

## Gotchas & tricks

- **Numerical stability.** Running the same block $N$ times accumulates norm drift; most implementations use QK-norm and careful residual scaling.
- **Attention over KV cache.** Caching $K$/$V$ per loop multiplies memory by $N$; some implementations cache only the last loop and recompute earlier ones.
- **Not a free depth multiplier.** On tasks that are well-fit by one forward pass, extra loops add compute without quality gains.
- **Halting is hard to train.** Adaptive-compute halting heads are notoriously unstable; many practitioners default to a fixed $N$ with LoopCD at decode time instead.
- **Intermediate-loop decodability** is not automatic — it requires either an auxiliary loss or the shared block to natively produce decodable states.

## Sources

- Paper: *Universal Transformers* — Dehghani et al., 2019 — the recurrent-block idea for sequence modeling.
- Paper: *Looped Transformers as Programmable Computers* — Giannou et al., 2023 — the computational-expressivity case for loops.
- Paper: *Decoding Looped Transformers Better for (Almost) Free* (LoopCD) — Liu et al., Apple, 2026 — [arXiv:2610.02185](https://arxiv.org/abs/2610.02185). Uses the Ouro and Huginn families as testbeds and introduces intermediate-loop contrastive decoding.
