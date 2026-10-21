# Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An inference-time trick to accelerate large-model decoding without changing its output distribution. A cheap *drafter* proposes $k$ future tokens; the *target* model verifies all $k$ in a single parallel forward pass; any consecutive accepted prefix is kept and the rest is thrown away. Because the forward is parallel, the throughput win approaches $k$ when the drafter's distribution matches the target's. Exact-equivalent variants exist; most production stacks relax to "mostly equivalent".

**Prereqs:** [attention](../fundamentals/attention.md), [mtp](../pre-training/mtp.md)
**Related:** [diffusion-drafted-speculative-decoding](diffusion-drafted-speculative-decoding.md)

---

## What it is

LLM inference is bottlenecked by *per-token memory bandwidth*: each decoded token loads the full model weights (and the KV cache) from HBM, but does trivially small compute. Speculative decoding fills that compute gap by verifying a block of candidate tokens per forward pass.

A speculative-decoding system has three parts: (1) a **drafter** that proposes $k$ tokens cheaply, (2) a **verifier** (the target model) that scores all $k$ in parallel, (3) an **acceptance rule** that decides which prefix to keep. The drafter can be a smaller LLM (classical), an extra set of heads attached to the target model (Medusa, EAGLE), an N-gram model, or — recently — a diffusion model.

## How it works

Decoding loop at step $t$:

1. **Draft.** The drafter emits a sequence $\tilde{y}_{t+1}, \tilde{y}_{t+2}, \dots, \tilde{y}_{t+k}$ with per-token probabilities $q_i$.
2. **Verify in parallel.** The target model runs one forward pass over the context + the $k$ drafted tokens, producing target probabilities $p_i$ at every drafted position *simultaneously* (this is cheap on GPU because the per-token FLOPs were already dwarfed by weight-loading).
3. **Accept.** For each $i$ in order, accept with probability $\min(1, p_i / q_i)$. The first $i^\star$ where this fails is the stopping point; sample position $i^\star$ from a corrected distribution $(p - q)_+$ to recover an exact-equivalence sample.
4. **Commit the prefix.** Keep $\tilde{y}_{t+1 \dots i^\star-1}$ plus the resampled $y_{i^\star}$. Advance $t \mathrel{{+}{=}} i^\star$ and go to step 1.

The expected acceptance length depends on $\mathrm{KL}(q \| p)$. For a well-matched 7B drafter on a 70B target, 3–5 tokens per round is typical.

Variants differ in how the drafter is realised:
- **Classical:** small assistant LLM. Reusable across many targets.
- **Medusa:** add $k$ independent prediction heads on top of the frozen target model. Shares the backbone — zero extra pass for draft.
- **EAGLE:** regress future hidden states (not tokens) with a small autoregressive head over the target's layer-2 features. Highest acceptance rates among AR drafters.
- **N-gram / lookup:** cheap but limited acceptance length; useful for low-entropy continuations like code boilerplate.
- **Diffusion drafter (DScout, 2026):** proposes many tokens *in parallel* via masked diffusion. See [diffusion-drafted-speculative-decoding](diffusion-drafted-speculative-decoding.md).

## Why it matters

- **No distribution change (in the exact variant).** Unlike greedy or temperature tricks, speculative decoding doesn't degrade quality — it's a pure throughput optimisation.
- **Compute-bound at last.** Modern GPUs are grossly underutilised during autoregressive decoding (~1–5% ALU utilisation). Speculative decoding pulls the arithmetic intensity up toward the hardware's natural operating point.
- **Stackable with batching, paged attention, FP8.** The win multiplies with other inference-stack improvements rather than competing with them.
- **Open-source parity.** Medusa / EAGLE / vLLM speculation have made 1.5–3× end-to-end speedups table-stakes for OSS serving of 70B-class models.

## Gotchas & tricks

- **Acceptance rate ≠ speedup.** If the per-round latency (drafter + verify) grows faster than the extra tokens accepted, throughput can drop. Tune $k$ and drafter depth jointly.
- **Draft divergence under RLHF.** A drafter trained on raw web text verifies badly against an RLHF'd target because post-training moves the distribution. Fine-tune the drafter on the same post-training data.
- **KV cache for the drafter adds up.** Separate drafter means a second KV cache; attached-head drafters (Medusa / EAGLE) avoid this.
- **Rejection cost.** Rejected drafted tokens still consume PCIe bandwidth and verify compute. The "worst case" is sampling with high temperature, where acceptance drops to 1 token per round.
- **Exact vs lossy acceptance.** Production stacks often skip the resample-on-rejection step to simplify kernels; the resulting distribution differs slightly from the target. Users rarely notice; auditors sometimes do.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias, 2022 — the original formulation and acceptance proof.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al. (DeepMind), 2023 — concurrent, generalised.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024.
- Paper: *Factorization-Guided Training and Inference for Diffusion-Drafted Speculative Decoding* — Fan et al. (ByteDance), 2026 — covered in [diffusion-drafted-speculative-decoding](diffusion-drafted-speculative-decoding.md).
