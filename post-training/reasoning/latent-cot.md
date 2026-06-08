# Latent Chain-of-Thought
*Depth — reasoning in continuous "thought" vectors instead of (or alongside) discrete tokens.*

**TL;DR:** Textual CoT spends a token per reasoning step and forces every intermediate computation through the LM softmax. Latent CoT performs intermediate computation in **continuous vectors** that flow through the same causal stream as text. The current open question is how to keep the things that make textual CoT work — left-to-right generation, sampling, KV-cache decoding, tractable likelihoods — once the thoughts are continuous. NF-CoT (Fu et al., 2026) is the first recipe that recovers all four by parameterizing thoughts with a normalizing flow head spliced inside the LLM.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [grpo](../grpo.md)
**Related:** [length-penalty](length-penalty.md), [long2short](long2short.md), [rlvr](../rlvr.md)

---

## What it is

A family of recipes where the LLM emits **continuous-thought vectors** at designated positions of the response, instead of (or interleaved with) text tokens. The thoughts never get verbalized; they only feed back into the residual stream of the next layer/step. This trades the discrete-token serialization cost for higher-bandwidth latent computation — at the price of losing exact likelihoods and easy sampling unless the latent head is carefully chosen.

## How it works

The shared mechanism:

1. **Two heads, one stream.** At each generation step, the model decides whether the next position is a "thought" or a "text" position. Text positions go through the usual LM softmax. Thought positions emit a continuous vector through a *latent head*.
2. **Latent head must be tractable to be useful.** Naïve choices (a linear projection to a fixed-dim vector) break sampling and likelihoods. NF-CoT uses a **TARFlow-style normalizing flow**, which gives:
   - Exact density over latent thoughts (needed for RL post-training).
   - Sampling (needed for left-to-right decoding).
   - Compatibility with the existing **KV cache** (the latent head reads the same hidden state the LM head would).
3. **Distillation from textual CoT.** The latent flow is trained to compress explicit chain-of-thought traces into the continuous space, so the structure of the reasoning survives the format change.
4. **RL is the natural next step.** Because the flow gives a tractable likelihood over thoughts, you can directly run policy gradients (GRPO-style) on the latent reasoning space — something the older non-tractable latent-CoT proposals couldn't do.

## Why it matters

- **Bandwidth.** A single continuous thought-vector carries more bits than a discrete token. For reasoning whose intermediate state is fuzzy or partially formed, you skip the verbalization tax.
- **Inference cost.** Fewer reasoning tokens means fewer KV-cache extensions and fewer attention ops. NF-CoT reports lower intermediate-reasoning cost vs. explicit CoT on code generation while improving pass rates.
- **Compatibility.** The TARFlow-inside-the-backbone construction is the first that preserves *all* of left-to-right generation, sampling, KV-cache, and tractable likelihood at once. That makes latent reasoning a plausible drop-in for production inference, not just a research artifact.

## Gotchas & tricks

- **The thought distribution can collapse.** If the flow's prior is too tight, the latent positions become near-deterministic and the bandwidth advantage disappears. Distillation from diverse CoT traces helps preserve entropy.
- **Decoding policy.** You still need to decide *when* to emit a thought vs. a text token. Fixed schedules (one thought per N text) are simpler but less expressive than learned gates.
- **Eval ambiguity.** Pass rate is easy to compare against textual CoT; *what the model thought* isn't directly inspectable any more. Probing the residual stream becomes the main interpretability lever.
- **RL stability.** Latent space policy gradients work in principle, but the variance is often higher than text-space PG. Normalize advantages within the latent action carefully.

## Sources

- Paper: *Latent Reasoning with Normalizing Flows* — Fu et al., UPenn / UCSD / Meta, 2026 — [arXiv:2606.06447](https://arxiv.org/abs/2606.06447) — NF-CoT, the first latent-reasoning recipe compatible with the modern inference stack.
- Related: TARFlow / autoregressive normalizing flows — the latent-head primitive used by NF-CoT.
