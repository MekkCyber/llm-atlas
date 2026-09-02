# Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Lossless LLM inference speedup that decouples token *proposal* from token *verification*. A small, cheap **draft model** proposes $k$ next tokens; the large **target model** verifies all $k$ in a single forward pass and keeps the longest accepted prefix. Because most easy next-tokens are ones the small model already agrees with, the target model gets amortized over multiple tokens per forward — 2–3× wall-clock speedup at zero output-distribution change.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [verification-aware-training.md](verification-aware-training.md)

---

## What it is

Autoregressive decoding is memory-bandwidth-bound: each new token needs one full forward pass of the LLM, but each pass reads the entire weights matrix to produce a single scalar-per-vocab distribution. If we could get multiple tokens out per forward, we'd amortize the bandwidth.

Speculative decoding does exactly that using rejection sampling: a fast draft $q$ guesses $k$ tokens ahead, the slow target $p$ scores all $k$ in one pass, and we accept a prefix distributed *identically* to sampling from $p$ alone. Output distribution is preserved exactly — no quality loss.

## How it works

At decode step $t$:

1. **Draft.** Run draft $q$ to sample $k$ candidate tokens $x_{t+1}, \ldots, x_{t+k}$, recording $q(x_{t+i} \mid x_{<t+i})$ for each.
2. **Verify (one target forward).** Run target $p$ once on the prefix $+$ candidates. This yields $p(\cdot \mid x_{<t+i})$ for every position $i \in [1, k]$ in a single pass thanks to the causal mask.
3. **Rejection-sample position by position.** For each $i$, accept $x_{t+i}$ with probability $\min(1, p(x_{t+i}) / q(x_{t+i}))$. On first rejection at position $j$, resample from the *residual* distribution $\max(0, p - q)$, normalized, and stop.
4. **Bonus token.** If all $k$ candidates accept, sample one extra token from $p$ at position $k+1$ — the target's own final-position logits are already computed.

Expected accepted length per outer step is $\approx k \cdot \alpha$, where $\alpha \in [0, 1]$ is the mean per-position acceptance rate.

## Why it matters

- **Zero-quality-loss speedup.** Same output distribution as pure target sampling; no calibration, no eval drift.
- **Bandwidth arithmetic.** Target forward cost dominates draft cost by ~$100\times$; one verify pass yields $\approx k\alpha$ tokens. For $\alpha = 0.7$, $k = 5$: 3.5× throughput.
- **Composes with paged attention, continuous batching.** No change to the target model or KV cache layout beyond scoring $k$ extra positions.
- **Foundation for a family.** Medusa (parallel heads), EAGLE (feature-level draft), Lookahead decoding, tree-attention variants all inherit the accept/reject core.

## Gotchas & tricks

- **Draft quality is the whole game.** $\alpha$ drops sharply when draft and target diverge — different tokenizers, different fine-tuning distributions, or greedy vs sampling temperature mismatch. VAT ([verification-aware-training.md](verification-aware-training.md)) trains drafts *for verification*.
- **KV cache waste on rejection.** Target forward computed logits for positions past the first rejection; those KV entries must be discarded. Some implementations retain them for a resumed draft.
- **Batching interacts badly with variable acceptance.** Different requests in a batch accept different lengths — some tricks (packed prefill, tree attention) recover throughput.
- **Rejection at $i=1$ is the failure mode to watch.** Wastes the entire verification pass. Monitor first-position acceptance separately.
- **Greedy target ≠ speculative-greedy.** With temperature 0 on the target, accept/reject reduces to strict argmax equality — much stricter than the sampling case. Use temperature $>0$ or switch to strict deterministic-match speculative decoding.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., Google, 2022 — arxiv.org/abs/2211.17192.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., DeepMind, 2023 — arxiv.org/abs/2302.01318.
- Downstream: Medusa, EAGLE, Lookahead — all extend the draft/verify pattern.
