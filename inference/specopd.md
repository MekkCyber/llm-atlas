# SpecOPD — Branch-Parallel On-Policy Distillation for Speculative Decoding
*Depth — a parallel on-policy training recipe for drafters, optimizing acceptance length directly.*

**TL;DR:** Training a draft model for speculative decoding via on-policy distillation is bottlenecked by **sequential rollouts** — you draft, verify, update, redraft. SpecOPD runs **several draft branches in parallel**, packs their proposals into shared **tree-structured prompts** that the target verifies in a single pass, and trains the drafter with an **E2E-TV surrogate** that targets *expected acceptance length* instead of per-token cross-entropy. Result: **+41.45% mean acceptance length**, **16.09× supervised-token throughput** over sequential-rollout OPD, and **up to 3.56× end-to-end generation speedup** over autoregressive decoding.

**Prereqs:** [README.md](./README.md) (speculative decoding basics)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

A drafter for speculative decoding (EAGLE / Medusa / Hydra lineage) is trained to **mimic the target model's next-token distribution**. On-policy distillation has become the dominant recipe — sample from the drafter, verify under the target, use the target's distribution as the training signal — but each training step is a sequential chain of draft/verify, which bounds the supervised-token throughput at training time.

SpecOPD breaks the sequential chain in two places: **parallel branches** of draft rollouts in each batch, and a **tree-packed verification pass** that scores all branches simultaneously through the target.

## How it works

**Branch-parallel sampling.** For each training prompt, the drafter spawns $B$ independent branches, each extending `k` tokens from the prompt. Branches diverge on sampled tokens but share the common prefix.

**Tree packing.** The $B \times k$ proposals are packed into a single shared proposal **tree** with a block-triangular attention mask so the target model verifies all branches in one forward pass. This is the same "token tree" idea as EAGLE's draft trees, but used at *training* time to amortize the target cost across branches instead of at inference.

**E2E-TV surrogate loss.** Instead of per-token cross-entropy against the target distribution, SpecOPD trains against a surrogate that directly estimates the **expected end-to-end acceptance length** — the expected number of tokens the target would accept in one speculative round. The surrogate factorizes acceptance probabilities per depth and is differentiable w.r.t. drafter logits.

**On-policy loop.** Drafter is updated each step from its own samples, verified by the frozen target. The branches re-sample every iteration — this keeps the training distribution aligned with the drafter's evolving sampling distribution.

## Why it matters

Drafter training cost has been the gating factor for shipping EAGLE / Medusa drafters per-model in production: training sequentially takes GPU-weeks per base model. A 16× supervised-token-throughput win makes it practical to retrain drafters per release. The acceptance-length-targeting surrogate is also conceptually right — per-token cross-entropy is a proxy, acceptance length is the quantity you ship.

## Gotchas & tricks

- Tree packing complicates the target model's attention mask — upstream serving infra needs to support arbitrary tree masks (EAGLE's tree masks work).
- The E2E-TV surrogate requires the target's token distribution, not just top-1. Save the target's top-k logits during verification to feed the surrogate.
- Branch count $B$ trades training throughput against sample diversity — too many branches sharing a prefix oversample the same region. $B = 4\!-\!8$ is typical.
- Still verifies against the *target's* distribution at serving time; SpecOPD does not change correctness guarantees.

## Sources

- Paper: *SpecOPD: Branch-Parallel On-Policy Distillation for Speculative Decoding* — Liu et al., 2026 — https://arxiv.org/abs/2611.02846
- Prior: *EAGLE / EAGLE-2*, *Medusa* — draft-tree speculative decoding.
