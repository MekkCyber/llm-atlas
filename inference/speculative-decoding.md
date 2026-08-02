# Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Have a small, cheap **draft model** propose the next few tokens; have the large **target model** verify them in parallel with a single forward pass. Accepted drafts are kept; the first rejection triggers a resample and the process restarts. When the draft is well-aligned, one target step produces multiple output tokens — a ~1.5–3× decoding speedup with **exactly the same output distribution** as the target model.

**Prereqs:** [transformer-block](../architectures/transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [mtp](../pre-training/mtp.md), [lossy-verification](lossy-verification.md)

---

## What it is

Autoregressive decoding is sequential — one token per target forward pass, which is memory-bandwidth bound. Speculative decoding turns that into batched verification: a draft model proposes $k$ candidate tokens; the target computes probabilities for all $k$ positions in a single pass; a per-token acceptance test either keeps the draft token or resamples from a corrected distribution. Because verification is exact (rejection-sampling based), the output distribution is provably identical to sampling from the target directly.

## How it works

Given target $p$ and draft $q$, for each drafted token $x_t \sim q(\cdot|x_{<t})$:

- Accept with probability $\min\bigl(1, \tfrac{p(x_t|x_{<t})}{q(x_t|x_{<t})}\bigr)$.
- On rejection, resample from the **residual** distribution $\mathrm{norm}(\max(0, p - q))$ and stop.

If all $k$ drafts are accepted, an extra bonus token is sampled from $p$ at position $k{+}1$ — so one target pass emits between 1 and $k{+}1$ tokens.

Draft sources in practice:
1. **Separate small model** (Leviathan / Chen 2023): a distilled or trained lightweight LM sharing the target's tokenizer.
2. **Self-drafting head** (Medusa, EAGLE, [MTP](../pre-training/mtp.md)): extra heads on the target model itself predicting tokens $t{+}1 \ldots t{+}k$.
3. **Tree drafting** (EAGLE-2, SpecInfer): draft multiple candidate continuations simultaneously and verify the tree in one pass — increases acceptance at the cost of a wider verify kernel.

## Why it matters

Autoregressive decoding is the dominant serving cost. SD unlocks a ~2× throughput gain at fixed model quality with a small memory overhead (the draft) and negligible engineering complexity in modern serving stacks (vLLM, SGLang, TensorRT-LLM ship SD out of the box). It's the primary reason MTP-style draft heads are added at pretraining time.

## Gotchas & tricks

- **Tokenizer alignment.** Draft and target must share a tokenizer, otherwise no token-level probability comparison is possible.
- **Acceptance rate is everything.** A draft that agrees with the target 30% of the time barely helps; ~70%+ is where the wins live. Better drafts (bigger, distilled from target, or trained MTP heads) pay for themselves.
- **Batch interaction.** SD reduces per-request latency but complicates continuous batching — accepted-length variance across requests fragments the batch. Modern schedulers handle this but naive integrations regress throughput.
- **Lossy variants are not free.** Truncation-based and collaborative "faster" SD variants silently rewrite the output distribution — see [lossy-verification](lossy-verification.md) for the failure modes.
- **Temperature ≠ greedy.** At $T=0$ (greedy target), acceptance simplifies to a strict argmax match — draft-target disagreement kills all gains. Temperature-sampled decoding is where SD shines.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias, 2023 — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al. (DeepMind), 2023 — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318).
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE / EAGLE-2* — Li et al., 2024 — feature-based draft heads with tree verification.
- Paper: *Revisiting Lossy Verification in Speculative Decoding* — Wang et al., 2026 — analysis of lossy SD variants and their failure modes.
