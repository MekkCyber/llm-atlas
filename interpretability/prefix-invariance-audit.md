# Prefix-invariance audit
*Depth — a gradient-free, two-forward-pass audit that localizes causality violations in sequence models.*

**TL;DR:** Causal sequence models are supposed to guarantee **prefix invariance**: the representation at position `t` must not depend on any input at position `> t`. In practice, hybrid attention/SSM stacks can leak future information through normalization, cache reuse, or a mis-implemented mixer — even when the attention mask looks correct. This audit runs two forward passes (one with the suffix, one with the suffix perturbed) and produces a per-layer score that localizes where causality breaks.

**Prereqs:** [../architectures/multi-head-attention](../architectures/multi-head-attention.md), [../architectures/transformer-block](../architectures/transformer-block.md)
**Related:** [../architectures/mla](../architectures/mla.md)

---

## What it is

A cheap post-hoc audit distinguishing *causal intent* (the mask) from *causal behavior* (the actual dependency graph). Runs on any sequence model without training, gradients, or architectural changes; returns a per-layer score that flags leakage.

## How it works

- Take a batch of sequences.
- **Pass 1:** run the model normally; cache the layer outputs at each position `t`.
- **Pass 2:** perturb only the *suffix* (positions `> t` for the target prefix length); re-run and cache outputs.
- For each layer, measure the change in the prefix-position representations between pass 1 and pass 2. Under strict prefix invariance the change is exactly zero.
- Aggregate the per-position, per-layer deltas into a **per-layer causality score**. Non-zero scores localize the layer where future information leaks in.
- Repeat with several perturbation types (random tokens, adversarial tokens) to distinguish deterministic leakage from numerical noise.

## Why it matters

Causality bugs quietly inflate train and eval numbers (the model peeks at future tokens during training) and can silently break generation (train-time behavior doesn't match single-token decoding). Existing tests inspect the attention mask; this audit inspects the actual computation, so it catches leakage that mask inspection misses — normalization across positions, KV-cache misuse, hybrid mixer bugs.

## Gotchas & tricks

- Numerical noise can look like a small non-zero score. Pick a threshold empirically per model; validate on a known-clean baseline.
- Perturbing tokens vs perturbing embeddings gives different sensitivities — token perturbations catch tokenizer/embedding-side bugs, embedding perturbations catch mixer bugs.
- Especially useful for **hybrid attention/SSM stacks**, where the leakage surface is bigger than attention alone.
- Cheap enough to run on every commit that touches a mixer implementation.

## Sources

- Paper: *The Mask Is Not the Model: Auditing Prefix Invariance in Attention, State-Space, and Hybrid Sequence Models* — Kim et al., 2026 — [arXiv:2608.22876](https://arxiv.org/abs/2608.22876)
