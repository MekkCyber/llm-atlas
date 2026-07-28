# Molt
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A PyTorch-native training framework for agentic RL, built to keep researcher iteration cost small. Molt runs a **single asynchronous loop** that generates rollouts and trains simultaneously, while enforcing that every optimized token was produced by the current policy version (token/policy/semantics consistency). Trains multimodal and MoE policies end-to-end, and reports parity with a Megatron-based state-of-the-art stack under a matched async protocol. Released by NVIDIA.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [ray.md](./ray.md), [partial-rollouts.md](./partial-rollouts.md), [dualpipe.md](./dualpipe.md)

---

## What it is

An RL training framework whose design axis is *code compactness for research iteration*, not feature-completeness. The agent is written as an ordinary Python program (one loop that alternates generation and update) instead of a trainer/rollout/backend graph. Every RL algorithm change is a local edit, not a change threaded across framework layers.

## How it works

- **One async loop.** Rollouts and gradient updates run concurrently in the same process. A rollout submits generated experiences into a queue; the trainer pulls, filters, and updates.
- **Strict on-policy filter.** Each experience is tagged with the policy version that produced it. Any experience whose version doesn't match the current trainer version is discarded before the loss is computed — the trainer never sees off-policy tokens even though rollouts and updates overlap in wall time.
- **Multimodal / MoE support.** The single loop treats these as ordinary model classes; sharding and expert placement live in ordinary PyTorch (FSDP / expert parallel), not framework-specific abstractions.
- **Matched-protocol comparison.** Under an identical fully-async protocol, Molt is statistically comparable to a Megatron-based reference stack in reward and throughput.

## Why it matters

Established RL frameworks (veRL, TRL, OpenRLHF, NeMo-Aligner) have grown into layered systems where an algorithm tweak — a new advantage estimator, a new rollout policy — touches trainer, distributed backend, and rollout glue. Molt argues the *right* tradeoff for research code is one an AI assistant can hold in its context window, even at the cost of some plumbing generality. It gives the field a compact open baseline for async on-policy agentic RL.

## Gotchas & tricks

- **Strict on-policy filter is the whole trick.** Weaken it (allow one-version-old experiences) and you re-introduce the correctness bugs Molt was designed to avoid.
- **Compactness ≠ minimalism.** MoE and multimodal support are in scope; feature-completeness for every RL variant is not.
- **Not a drop-in for Megatron-scale runs.** Parity is *statistical*, matched-protocol; production Megatron pipelines still cover cases (custom kernels, exotic parallelism) that a lean framework does not.

## Sources

- Paper: *Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning* — NVIDIA (Li, Zhang, Xu, Zhang, Zhang, Desai, Demoret, Molchanov, Kautz, Dong), 2026 — [arXiv:2607.21653](https://arxiv.org/abs/2607.21653).
- Code: [github.com/NVIDIA-NeMo/labs-molt](https://github.com/NVIDIA-NeMo/labs-molt)
