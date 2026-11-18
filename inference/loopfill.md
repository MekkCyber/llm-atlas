# LoopFill
*Depth — pipelined dual-axis speculative decoding for looped Transformers.*

**TL;DR:** LoopFill is a training-free speculative decoding algorithm for **looped** LLMs — models that apply the same transformer block *k* times per output token (Ouro, Huginn, energy-based reasoners). A standard draft-and-verify scheme only speculates in the token dimension, wasting the fact that most tokens commit after far fewer loops than the maximum *k*. LoopFill builds a single draft tree that mixes **multi-token suffix drafts** with **shallow-loop predictions**, then verifies all branches in one pipelined pass. Short-loop drafts short-circuit deep loops when they agree.

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md), [../pre-training/mtp.md](../pre-training/mtp.md)
**Related:** [../post-training/reasoning/energy-based-reasoner.md](../post-training/reasoning/energy-based-reasoner.md), [../post-training/reasoning/README.md](../post-training/reasoning/README.md)

---

## What it is

Looped LLMs are the recurrent-in-depth cousin of long-CoT reasoners: instead of spending tokens on scratch work, they apply the same transformer block *k* times before emitting each token. This gives test-time compute scaling without CoT verbosity, but it multiplies decode latency by *k*. Prior speculative decoding for loops (LoopAhead) speculates only on the token axis; LoopFill observes that the **loop depth** itself is a speculation axis — most tokens would be correct if the model stopped at loop *j* < *k*.

## How it works

Three moves stacked:

1. **Dual-axis draft tree.** For each anchor position, build a tree whose *width* enumerates candidate next tokens (multi-token suffix drafts) and whose *depth* enumerates predicted outputs *after loop j* for *j* = 1 … *k*. Each node is a (token, loop-count) pair.

2. **Pipelined verification.** Shuffle the tree into one forward-pass batch so that verifying all nodes costs essentially the same as a single multi-token verify. The pipelining fuses the loop iterations with the batched candidates.

3. **Commit the longest prefix.** Accept the longest sequence of (token, loop-count) pairs along a tree path whose intermediate loop outputs match the full-loop target. Where a shallow loop *j* already matches what loop *k* would produce, commit that token at depth *j* and skip the remaining *k − j* loops.

No training is required; LoopFill is a pure decoding-time reorganization of the loops the model already runs.

## Why it matters

Looped reasoners (Ouro, Huginn) and energy-based reasoners are gaining ground as a compute-scaling alternative to long-CoT RL. Their decode latency has been the main objection. LoopFill reports **5.85× speedup on Ouro-2.6B** and **5.13× on Huginn-3.5B** over greedy autoregressive decoding, exceeding the single-axis LoopAhead baseline by **40%** and **25%** respectively. That removes the main inference-time penalty of looped reasoners and makes them more competitive with CoT-at-inference setups.

## Gotchas & tricks

- The acceptance criterion requires the intermediate loop state to be comparable across depths; models that mix state meaningfully only at the final loop block will see low acceptance.
- The speedup depends on the *distribution of effective loops* per token. Workloads that always need the full *k* (e.g., hardest Sudoku states for energy-based reasoners) see smaller gains.
- The batched verify pass is memory-heavier than vanilla decoding — the tree expands K-fold. On long contexts this fights with KV-cache budget; pair with FP4/FP8 KV cache quantization if memory is tight.
- Combine with continuous batching: the dual-axis tree is cheap to interleave across requests since it is already shaped as a batch.

## Sources

- Paper: *LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers* — Siqi Zhu, Suozhi Huang, Yi Zhao, Jiaxuan You, 2026 — [arXiv:2611.19649](https://arxiv.org/abs/2611.19649).
- Context: Ouro and Huginn — the looped reasoning model families LoopFill benchmarks against.
