# Looped / Recurrent-Depth Transformer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A transformer that shares one (or a few) blocks across many *recurrent passes* instead of stacking distinct blocks. Parameter count stays small; effective depth grows by looping the same weights $L$ times. The approach has resurfaced in 2025–2026 (Ouro, Huginn) as a way to buy reasoning depth at near-constant parameter count, and admits new decoding tricks that exploit the aligned weak/strong pairs produced by intermediate loops.

**Prereqs:** [transformer-block.md](transformer-block.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../inference/contrastive-decoding.md](../inference/contrastive-decoding.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Standard transformers are deep stacks of unique blocks: $N$ layers, $N$ sets of weights, cost scales with $N$. Looped (recurrent-depth) transformers share one block (or a short cycle of blocks) and apply it $L$ times to the hidden state:

$$
h_{t+1} = \mathrm{Block}_\theta(h_t) \qquad t = 0, 1, \ldots, L{-}1
$$

Parameter count is independent of $L$. Each loop is a forward pass of the same weights; FLOPs scale with $L$. Variants:

- **Fully-shared** (Universal Transformer, 2018): one block, applied $L$ times.
- **Cycle-shared** (modern Ouro-style): a short cycle of 2–4 distinct blocks, repeated $L / k$ times.
- **Dynamic halting** (Universal Transformer, PonderNet): learn to stop looping per position when confident.

Modern training treats $L$ as a hyperparameter at train and inference time independently; "think longer" means more loops.

## How it works

- Shared block typically has pre-norm, attention + FFN just like a standard transformer block.
- Loop-count signal: either a scalar embedding of $t$ added at each step, or no signal at all (letting the hidden state carry its own position-in-loop information).
- Training: either fix $L$ or sample $L$ from a distribution each step. Supervise the output after the *final* loop (plus optionally after intermediate loops as auxiliary losses).
- Inference: pick $L$ for your compute budget. Larger $L$ generally improves quality on reasoning tasks up to a saturation point.

## Why it matters

- **Reasoning depth without parameters.** A ~2–3B looped model can reach the reasoning depth of a much larger standard transformer at a fraction of memory cost — useful on-device and in low-memory inference.
- **Decoding-time compute lever.** Unlike a standard transformer where depth is baked in, loop count is a user-tunable knob at inference.
- **Natural weak/strong pairs.** Intermediate loop outputs are *weak* versions of the final prediction, aligned with the strong one by construction. This is what makes LoopCD work (see [../inference/contrastive-decoding.md](../inference/contrastive-decoding.md)): a free contrastive-decoding signal without a second model.

## Gotchas & tricks

- **Diminishing returns on $L$.** Most tasks saturate well before $L=16$. Picking $L$ at inference is a tuning job per deployment.
- **KV cache cost.** Each loop re-attends over the full prefix. For long contexts, loops multiply attention FLOPs — the memory-saving advantage shrinks.
- **Pre-norm placement matters.** Shared blocks accumulate residual drift across loops; pre-norm with careful init is standard.
- **Not the same as "deep thinking" tokens.** Looped depth is a *fixed-time-cost per token* lever; chain-of-thought is a token-count lever. The two stack.
- **Training stability tricky.** The gradient path through $L$ loops is $L$ layers deep *of the same weights*; too-small $L$ at training, too-large $L$ at inference, generalizes poorly. Train with the inference $L$ distribution in mind.

## Sources

- Paper: *Universal Transformers* — Dehghani et al., 2018 — original recurrent-depth transformer with dynamic halting.
- Paper: *PonderNet: Learning to Ponder* — Banino et al., DeepMind, 2021 — learnable halting policy.
- Paper: *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent-Depth Approach* (Huginn) — Geiping et al., 2025 — modern 3.5B recurrent-depth model.
- Paper: *Ouro: Looped Transformer Reasoners* — 2026 — 2.6B-class looped model with thinking modes.
- Paper: *Decoding Looped Transformers Better for (Almost) Free (LoopCD)* — Liu et al., Apple, 2026 — [arXiv:2610.02185](https://arxiv.org/abs/2610.02185).
