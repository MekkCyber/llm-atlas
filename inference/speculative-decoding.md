# Speculative decoding
*Depth — accelerating LM decoding by proposing multiple tokens with a small draft model and verifying with the target.*

**TL;DR:** A small "draft" model generates *k* candidate tokens; the "target" (big) model scores all *k* in one forward pass and accepts the longest prefix consistent with its own distribution. If acceptance is high, wall-clock decoding drops toward the draft's speed while staying **exactly** distributed as the target.

**Prereqs:** [attention.md](../fundamentals/attention.md), [transformer-block.md](../architectures/transformer-block.md)
**Related:** [_kv-cache-compression.md](_kv-cache-compression.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

An exact acceleration for autoregressive decoding. Target latency per token is bounded below by one target forward pass; specdec breaks that by amortising *k* accepted tokens across a single target pass, at the cost of running a cheap draft model in parallel.

## How it works

Per decoding step:

1. **Draft:** run the draft model *k* steps autoregressively to produce candidate tokens `x̂_{1..k}` with probs `q(·|·)`.
2. **Verify:** run one target forward on the *k*-token candidate to get target probs `p(·|·)` at each position.
3. **Accept/reject** each candidate token `x̂_i` with probability `min(1, p_i / q_i)`. Stop at the first rejection.
4. On rejection, sample the replacement token from the residual distribution `max(0, p − q) / normaliser`. Advance by (accepted + 1) tokens; goto 1.

Because the accept-reject rule is a bias-free importance-sampling step, the joint output distribution equals the target's — no quality loss. Speedup = (expected accepted tokens per round) × (target-pass cost) / (draft cost + verify cost).

Common draft choices:
- **Smaller same-family model** (e.g. Llama 7B drafts for Llama 70B).
- **Self-drafting**: EAGLE / Medusa attach lightweight heads to the target to predict the next few tokens without a separate model.
- **N-gram / retrieval** drafts for prompt-heavy workloads.

## Why it matters

- **Same output distribution.** Unlike quantisation or pruning, specdec doesn't trade quality for latency.
- **Compounds with batching.** In continuous-batching servers (vLLM, SGLang) draft-model cost amortises across concurrent requests.
- **Dominant lever for reasoning inference.** Long CoT traces are decoding-bound; drafts with 60–80% acceptance easily 2–3× throughput.
- **RL rollouts.** Rollouts dominate RL post-training cost — specdec at scale is now standard in frontier RL pipelines, with online co-training of the draft to keep acceptance high as the policy drifts (Wang et al., 2026).

## Gotchas & tricks

- **Draft/target mismatch** kills acceptance; keep tokenisers identical and draft trained on similar data as target.
- **Long context** breaks naive drafts — the draft's KV cache grows with the target's; use compressed draft caches or restart drafts.
- **Temperature > 0** hurts acceptance super-linearly; specdec is most useful at low-temperature inference.
- **Speculative batching**: for k tokens on B requests, target does one forward on B×k tokens — memory grows accordingly; tune `k` per KV budget.
- **Systems friction at scale**: branch attention isn't supported under standard causal context-parallel implementations, and target intermediate features span pipeline stages — production runs need custom kernels + off-schedule feature transports (see NeMo-RL online-draft co-training).

## Sources

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* — 2022 — https://arxiv.org/abs/2211.17192
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* — 2023 — https://arxiv.org/abs/2302.01318
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — 2024 — https://arxiv.org/abs/2401.15077
- Wang et al., *Online Draft Co-Training for Speculative Decoding in Large-Scale, Long-Context RL Post-Training* — 2026 — https://arxiv.org/abs/2609.07108
