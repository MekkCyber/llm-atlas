# Speculative Decoding
*Depth — a draft/target pair that accelerates autoregressive generation while preserving the target model's output distribution.*

**TL;DR:** A small, fast **draft model** predicts $K$ candidate next tokens greedily; the target model scores all $K+1$ positions in **one parallel forward pass**; a **rejection-sampling** step accepts the longest valid prefix and samples any rejected token from a corrected distribution. The target's exact token distribution is preserved. Speedup grows with the draft's acceptance rate and is substantial for autoregressive-heavy workloads; speedup is limited by batching (which also batches the target) and by memory bandwidth.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [adversarial-speculative-reasoning.md](adversarial-speculative-reasoning.md)

---

## What it is

Autoregressive decoding generates one token per forward pass, so each step reads all model weights from HBM for one token of output — bandwidth bound, not compute bound. Speculative decoding *amortizes* those reads: a cheap draft model proposes several tokens; the target model verifies all of them with one parallel forward pass; most are accepted; the step produced ≥1 token for the cost of one target forward.

The key guarantee is **exactness**: the token distribution of the combined (draft + target + rejection-sampling) procedure is identical to the target's own sampling. No quality loss, only speed.

---

## How it works

Given a target model $p$ and a draft model $q$ (same vocabulary), at each step:

1. **Draft $K$ tokens.** Starting from the current prefix, sample $y_1, \ldots, y_K$ from the draft $q$, one at a time (cheap).
2. **Target scores in parallel.** Run one target forward pass on the entire $(prefix, y_1, \ldots, y_K)$ and obtain $p(\cdot \mid \ldots)$ at each of the $K+1$ positions.
3. **Rejection sampling at each position.** For $i = 1 \ldots K$:
   - Accept $y_i$ with probability $\min\bigl(1, p(y_i \mid \ldots) / q(y_i \mid \ldots)\bigr)$.
   - If rejected, **sample a replacement** from the **residual distribution** $p_{\text{resid}}(y) \propto \max\bigl(0, p(y) - q(y)\bigr)$, emit it, and stop.
4. **If all $K$ accepted**, additionally sample $y_{K+1}$ from $p(\cdot \mid \ldots)$ at the final position (free — already computed) and emit it.

Expected tokens per target forward pass is $\sum_{i=1}^{K+1} \prod_{j < i} \alpha_j$, where $\alpha_j$ is the per-position acceptance probability (the draft–target alignment rate). With $\alpha \approx 0.7$ and $K = 4$, that's ~2.5 tokens per target step — a ~2.5× speedup on a bandwidth-bound workload.

### Variants

- **Draft-target speculative** (Leviathan 2022, Chen 2023) — the baseline above; draft is a small LM.
- **EAGLE / Medusa** — self-speculative: the target model itself emits multiple-token drafts through added heads, eliminating the separate draft model.
- **Lookahead decoding** — draft from n-gram patterns in the target's own recent output; no auxiliary model.
- **Tree / batched speculation** — draft multiple branches in a tree and verify them all in one target pass; raises expected accepted length at the cost of per-step compute.

---

## Why it matters

- **Exact.** Preserves the target distribution. Unlike quantization or distillation, zero quality trade-off.
- **Single-request speedup is large.** 2–10× over autoregressive in typical setups. Bigger at high-draft-acceptance workloads (code, structured output).
- **Composes with KV caching and paged attention.** The target forward pass is a standard batched decode; no serving-stack rewrite.
- **Diminishes under heavy batching.** Large batches already saturate target throughput — the "free" parallel verification stops being free. See SpecPilot (2026) for deployment-specific optimization under batching.
- **A semantic surface, not just a speed one.** Which tokens come from draft vs target can be a design lever (see [adversarial-speculative-reasoning.md](adversarial-speculative-reasoning.md)).

---

## Gotchas & tricks

- **Vocabularies must match.** Draft and target tokenizers must agree exactly — otherwise acceptance probabilities and residual sampling are undefined.
- **Temperature rescaling.** For $T \ne 1$, both $p$ and $q$ must be temperature-adjusted before acceptance; otherwise the exactness guarantee fails.
- **Residual distribution is computed per position, not per token.** Implementations that reuse a cached residual across steps silently break exactness.
- **Acceptance rate drives everything.** A draft with 50% acceptance at $K=4$ saves less than one at 80% — a smaller faster draft often beats a larger slower one.
- **Memory doubles if the draft is a separate model.** EAGLE/Medusa-style self-speculation avoids this at the cost of a target-side architectural change.
- **Batching halves the single-request speedup roughly.** Deployment-specific tuning (draft length, batch regime, scheduler) matters — SpecPilot (2026) reports autoregressive→speculative speedups rising from 1.94× to 3.93× under batching through automated kernel discovery.

---

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias (Google), 2022, arXiv 2211.17192 — the original algorithm and proof.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al. (DeepMind), 2023, arXiv 2302.01318 — contemporaneous formulation with sampling-correction framing.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *SpecPilot: Autonomous Algorithm Discovery and Systems Optimization for Speculative Decoding* — Chai et al., 2026, arXiv 2611.02501 — deployment-specific tuning via autonomous agents.
