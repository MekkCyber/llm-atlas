# VIA-SD: Verification via Intra-Model Routing

*Depth — a multi-tier speculative decoding scheme that routes verification by token confidence: direct acceptance for high-confidence drafts, a slim intra-model verifier for medium-confidence drafts, full model only for low-confidence drafts.*

**TL;DR:** Standard speculative decoding pays the full-model verifier cost on every drafted token, even when the draft was obviously right. VIA-SD inserts a **confidence-gated routing step**: high-confidence draft tokens are accepted directly, medium-confidence tokens go through a slim verifier (an intra-model routing path within the same model — no separate checkpoint), and only low-confidence tokens hit the full model. **10–20% speedup over prior speculative-decoding baselines, 2.5–3× over non-drafting decoding.**

**Prereqs:** [_speculative-decoding](_speculative-decoding.md)
**Related:** [README](README.md)

---

## What it is

A three-tier variant of the standard draft-verify pipeline. Instead of {draft, full-verify}, VIA-SD has:

1. **Draft** (any draft mechanism — Medusa heads, EAGLE, small model, lookahead).
2. **Confidence gate** — the draft attaches a confidence score to each proposed token.
3. **Tiered verification**:
   - High confidence → direct acceptance, no verifier work.
   - Medium confidence → **slim verifier** (a cheap intra-model routing path).
   - Low confidence → full model verification.

The slim verifier is realized as a routing path *inside the same model* rather than a separate model: a subset of layers or a learned skip path that the model can run cheaply. Correctness is preserved by ensuring the slim verifier is a valid rejection-sampling proxy in the confidence regime where it's used.

## How it works

Per drafted token $\hat y_t$ with confidence $c_t$:

- If $c_t > \tau_{\text{high}}$: accept directly. Add $\hat y_t$ to the output and proceed.
- If $\tau_{\text{low}} < c_t \le \tau_{\text{high}}$: run the slim verifier on the prefix up to $\hat y_t$. Accept or reject under the standard speculative-decoding rejection rule, but using the slim verifier's distribution.
- If $c_t \le \tau_{\text{low}}$: run the full model on the prefix. Standard speculative verification.

The two thresholds $\tau_{\text{high}}, \tau_{\text{low}}$ are tuned for the target acceptance rate. The intra-model routing path is trained jointly with the main model (or distilled from it) so that the slim verifier's distribution is close enough to the full model's in the medium-confidence regime to give a useful acceptance rate.

The trick that makes this work in practice is that the slim verifier is **inside the same forward pass** as the full model — it isn't a separate checkpoint and doesn't add deployment complexity. It's a routing path the GPU can take when the confidence gate says so.

## Why it matters

- **Compute-aware verification.** Single-tier speculative decoding wastes the full-model verifier on tokens the draft was confident about. Tiering routes the right amount of compute to each draft.
- **Practical 10–20% gains over EAGLE/Medusa baselines** and 2.5–3× over plain decoding without giving up the correctness guarantee.
- **The "confidence-gated verification" framing generalizes.** Any pipeline with an acceptance step (RL critique, generative verifier, retrieval re-ranking) could similarly gate by confidence to skip expensive verification on easy cases.

## Gotchas & tricks

- **The slim verifier must be a good proxy in the medium-confidence regime.** If it disagrees with the full model on cases the gate routes to it, you trade quality for speed.
- **Confidence calibration of the draft matters more than draft accuracy.** A miscalibrated confidence gate routes hard cases to the cheap tier and easy cases to the expensive tier.
- **Threshold tuning is workload-specific.** Chat / instruction-following has different confidence distributions than code generation.
- **Implementation gotcha**: the intra-model routing path needs its own forward-pass scheduling — a naive implementation pays the full-model cost anyway because the slim path runs as a subset of the full forward.

## Sources

- Paper: VIA-SD — Xian et al. (2026) — [arXiv:2606.12243](https://arxiv.org/abs/2606.12243)
