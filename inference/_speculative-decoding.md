# Speculative decoding

*Taxonomy — inference-time techniques that draft multiple candidate tokens cheaply and let the target model verify them in parallel to reduce decoding steps.*

**TL;DR:** Autoregressive decoding is bottlenecked by the target model's forward pass per token. Speculative decoding lets a cheap "drafter" propose $k$ tokens in one shot, then the target model verifies all $k$ in parallel; accepted tokens skip a forward pass each. Variants differ on **who the drafter is** (separate small model, integrated head, self-mode of same model), **how draft tokens are proposed** (sequential AR, parallel, tree, diffusion), and **how verification length is chosen** (fixed, adaptive). Production 2026 default is a **learned parallel-or-hybrid drafter + adaptive verify** (DSpark, self-speculation).

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [dspark](dspark.md) · [self-speculation-decoding](self-speculation-decoding.md) · [confidence-scheduled-verification](confidence-scheduled-verification.md) · [mtp](../pre-training/mtp.md)

---

## The problem

Under standard AR decoding, generating $n$ tokens takes $n$ forward passes of the target model — a per-token latency wall independent of batch size. But most next tokens are "easy" (high-confidence completions the target would predict identically). If a cheap drafter can guess them, the target only needs one forward pass to verify a whole block, and easy tokens become nearly free.

Correctness constraint: the accepted output must match what pure AR sampling from the target would produce (in distribution, if temperature > 0). This is what makes speculative decoding lossless — it's a speedup, not an approximation.

## The shared pattern

Every variant does:

1. **Draft** $k$ candidate tokens cheaply.
2. **Verify** all $k$ with one forward pass of the target model.
3. **Accept** the longest prefix that matches (or matches under the rejection-sampling rule for stochastic decoding); resample the first mismatch.

They differ in drafter architecture, how the $k$ tokens are proposed, and how $k$ is chosen.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Separate small drafter (Leviathan, Chen) | run a tiny model AR to draft, target verifies | separate model to train & align distributions | when a well-aligned small model already exists |
| Medusa | multiple prediction heads on target, top-N per head form a tree | tree verification kernel complexity | drop-in acceleration of a fine-tuned target |
| EAGLE / EAGLE-2 | AR drafter shares target's hidden states via a lightweight head | needs joint training | best single-user speedup at small batch |
| [mtp](../pre-training/mtp.md) | multi-token prediction as a *pretraining objective*, head kept for drafting | drafter fixed to depth D from pretraining | when MTP is baked into pretraining anyway |
| Parallel drafters (MTP-K multi-head) | one shot proposes $K$ tokens with $K$ heads | suffix decay: no intra-block deps | when $K$ small, low concurrency |
| Semi-AR (DSpark) | parallel backbone + tiny sequential module for intra-block deps | more drafter compute | high-throughput serving with load variance |
| [self-speculation-decoding](self-speculation-decoding.md) | same model as drafter (diffusion mode) and verifier (AR mode) | requires tri-mode training | one-model deployment, no drafter alignment |
| Adaptive verify ([confidence-scheduled-verification](confidence-scheduled-verification.md)) | choose verification length $k^*$ per request | needs load-aware scheduling | high-concurrency production serving |

## How to choose

- **Training from scratch or pretraining underway** → bake in MTP (depth-1 is the sweet spot). Speculative decoding for free.
- **Fine-tuning existing target for drop-in acceleration** → Medusa or EAGLE-2. Small extra training, no new model.
- **Frontier production serving with load variance** → DSpark-style semi-AR drafter + confidence-scheduled verification. Best throughput-latency Pareto.
- **Single unified checkpoint, no drafter to manage** → self-speculation (Nemotron-Labs-Diffusion pattern). Diffusion drafts, AR verifies, same weights.
- **Rapid retrofit with no training budget** → separate small drafter from the same family (Llama-3-3B drafting Llama-3-70B).

Draft length, verify length, and tree width are all first-class hyperparameters; the trend is toward making them **adaptive per request**, not fixed globally.

## Adjacent but distinct

- **Continuous batching / paged attention** — orthogonal serving optimizations. Speculative decoding composes with them.
- **Draft-only parallel decoding** (Lookahead) — no verifier, uses n-gram cache. Non-lossless.
- **Diffusion LMs decoding independently** — not speculative; multiple tokens per pass but no verify step. Self-speculation *combines* both worlds.
- **Prefix caching / prompt caching** — pre-computed KV, not draft-verify. Different mechanism, different regime.

## Sources

- *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022 — original formulation.
- *Accelerating LLM Inference with Speculative Sampling* — Chen et al., DeepMind, 2023 — rejection-sampling correctness proof.
- *Medusa* — Cai et al., 2024 — post-hoc multi-head drafter.
- *EAGLE / EAGLE-2* — Li et al., 2024 — hidden-state-conditioned AR drafter.
- *DSpark* — Yu et al., DeepSeek, 2026 — semi-AR + confidence-scheduled verify — [arXiv:2607.05147](https://arxiv.org/abs/2607.05147).
- *Nemotron-Labs-Diffusion* — NVIDIA, 2026 — self-speculation — [arXiv:2607.05722](https://arxiv.org/abs/2607.05722).
