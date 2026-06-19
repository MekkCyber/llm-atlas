# Speculative Decoding

*Depth — accelerate autoregressive decoding by drafting cheaply and verifying in parallel.*

**TL;DR:** Replace one expensive sequential pass through a target LLM with a cheap **drafter** that proposes $k$ tokens, plus a single batched forward of the target that verifies them in parallel. Accepted tokens are kept; the first rejection rolls back to that position. Output distribution is **identical** to plain target-model sampling. Speedup depends on drafter quality and the regime — biggest wins are memory-bound serving where the target forward was paying for kv-cache and weight reads regardless of batch size.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [self-speculative-decoding](self-speculative-decoding.md)

---

## What it is

A serving-side latency reduction technique for autoregressive LLMs. Normally you decode one token per target-model forward pass; speculative decoding amortizes one target forward over multiple tokens by using a smaller / cheaper "drafter" to guess what they'll be.

## How it works

Per step:

1. Drafter $q$ proposes $k$ candidate tokens $\hat{o}_1, \ldots, \hat{o}_k$ autoregressively (cheap because the drafter is small).
2. Target $p$ does **one** parallel forward over the prompt + all $k$ candidates, producing $p(\cdot \mid \text{prefix})$ at each position.
3. For each position $j$ in order, accept $\hat{o}_j$ with probability $\min(1, p(\hat{o}_j) / q(\hat{o}_j))$.
4. On first rejection at position $j$, resample $o_j \sim \text{normalize}(\max(0, p - q))$ and stop. All accepted tokens before $j$ are committed.

This rejection-sampling scheme guarantees the marginal output distribution is exactly $p$ — speculative decoding does not change what the model says, only how fast it says it.

If all $k$ tokens are accepted you generated $k+1$ tokens (the $k$ accepted plus the bonus token from the target's own next-token prediction at position $k+1$) in one target forward; if all rejected you generated 1. Expected gain depends on the **acceptance rate** $\alpha$ and the drafter cost ratio.

## Why it matters

- **Memory-bound serving wins big.** Decoding small batches is gated by weight + KV-cache reads, not flops. The target forward over $k$ candidates costs roughly the same as a single forward — you get the extra tokens almost for free.
- **No quality loss.** Unlike distillation or quantization, speculative decoding is *exact* — the output distribution is the target's.
- **Composes with everything.** Works on top of vLLM, SGLang, paged attention, continuous batching, FP8 weights. Now standard in production LLM serving stacks.

## Gotchas & tricks

- **Compute-bound regimes don't benefit.** Large batches saturate flops; the target forward over $k$ candidates costs ~$k\times$ more in that regime, killing the win. Some servers turn SD off above a batch-size threshold.
- **Drafter–target mismatch kills $\alpha$.** Drafters trained on different data or distilled badly drop $\alpha$ to ~0.3, eating the speedup. Best drafters are distilled directly from the target's outputs.
- **$k$ is a tradeoff.** Bigger $k$ = more potential tokens per step but exponentially more sensitivity to a single rejection. $k = 4$–$8$ is typical.
- **Tree variants** (Medusa, EAGLE) draft a **tree** of candidates rather than a chain, so a rejection at position $j$ can still accept tokens from a different branch — higher amortization at the cost of a more complex verifier.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias (Google), 2022.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al. (DeepMind), 2023.
- Paper: *EAGLE* — Li et al., 2024 — extension with feature-level drafting + tree verification.
- Paper: *Medusa* — Cai et al., 2024 — multi-head drafter with tree verification.
