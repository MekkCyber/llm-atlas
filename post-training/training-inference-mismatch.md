# Training-inference mismatch
*Depth — the numeric gap between the engine that samples rollouts and the engine that computes gradients in LLM RL pipelines.*

**TL;DR:** In modern LLM-RL stacks (RLHF, GRPO, RLVR), rollouts are produced by an inference engine (vLLM, SGLang, TensorRT-LLM) while gradients are computed by a training engine (Megatron, FSDP, DeepSpeed). The two assign *different* probabilities to the same tokens for the same weights, breaking the on-policy assumption and inflating importance-sampling variance. This page covers where the gap comes from and how it is usually patched.

**Prereqs:** [rlvr](rlvr.md), [ppo](ppo.md)
**Related:** [calibrated-importance-sampling](calibrated-importance-sampling.md), [grpo](grpo.md), [_rl](_rl.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

A pervasive, quietly-tolerated source of bias in decoupled generate-then-train LLM RL loops. The two engines are supposed to be numerically equivalent; in practice they are not, and their per-token log-probabilities can differ by amounts that matter for importance-weighted policy gradients.

## How it works

Sources of the gap, roughly in order of impact:

- **Attention kernel differences.** FlashAttention variants, page sizes, KV-cache layouts, and the reduction order of the softmax normalizer all differ across engines. Even bit-for-bit identical weights produce slightly different logits.
- **MoE routing.** Router top-k decisions are unstable near ties; a single flipped expert can change the token's logit significantly. Inference engines often fuse the router differently from the training engine.
- **Precision.** BF16 vs FP16, FP8 activation quantization at inference, LayerNorm epsilons — small precision choices compound over many layers.
- **Temperature and sampling code paths.** Inference engines apply top-p / min-p filtering before recording logprobs, while the training engine sees the unfiltered logits.
- **Continuous batching.** Different batch composition changes RNG state and (for some kernels) numeric results.

At the level of policy updates, the effect is that the importance ratio `π_train(a|s) / π_infer(a|s)` deviates from 1 in a token-dependent way, and its second moment can be arbitrarily large without correction.

## Why it matters

If you use the inference-engine logprob as-is (`ratio = 1`, i.e. treat the rollouts as on-policy), you introduce a silent bias whose direction is hard to predict. If you use exact importance sampling, variance explodes on the tail of the ratio. Every serious LLM-RL system therefore lives on a spectrum between these two failure modes, patched with clips, temperature shifts, or bespoke kernel matching. This is one of the top three practical stability issues in RLVR training runs.

## Gotchas & tricks

- **Log both.** Whatever your correction is, record per-token `π_train` and `π_infer` on a validation batch every epoch — if the distribution of `δ = logit_train - logit_infer` shifts, something in your engine stack changed.
- **Router determinism.** For MoE, force deterministic top-k and matching capacity factors across engines; this alone kills a big fraction of the gap.
- **Don't chase bit-parity.** Trying to make the two engines produce identical logits is a losing battle; the fix is a good *estimator* (see [calibrated-importance-sampling](calibrated-importance-sampling.md)) rather than perfect matching.
- **Partial rollouts** ([../systems/partial-rollouts.md](../systems/partial-rollouts.md)) exacerbate the mismatch — rollouts started on one engine and continued on another see the gap twice.
- Temperature matters: at `T → 0` the mismatch collapses (arg-max is stable), which is why some teams unofficially "cheat" by RLing with a lower sampling temperature than they deploy at. This trades stability for train-serve drift.

## Sources

- Paper: Rethinking Training-Inference Mismatch in LLM Reinforcement Learning — Yu et al., 2026 — [arXiv:2609.32444](https://arxiv.org/abs/2609.32444)
- Practitioner reports: OpenRLHF and Verl issue trackers, DeepSeek-V3 and Qwen tech reports (partial-rollout discussions).
