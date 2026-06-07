# NF-CoT — Latent Reasoning with Normalizing Flows
*Depth — replace some chain-of-thought tokens with continuous latent thoughts modelled by a normalizing flow, while keeping autoregressive sampling, KV cache, and exact likelihoods.*

**TL;DR:** Textual CoT spends a token per reasoning step, even when the step is fuzzy or only half-formed. Prior "latent reasoning" methods compress reasoning into continuous vectors but sacrifice autoregressive guarantees: KV cache breaks, sampling becomes ad-hoc, likelihoods are intractable. **NF-CoT** (Fu et al., 2026) instantiates a TARFlow-style normalizing flow head inside the LLM backbone: at "thought" positions an NF head emits a continuous latent with a tractable log-likelihood; at text positions the standard LM head emits a token. Left-to-right, KV-cached, sampleable, and directly policy-gradient-trainable in the latent space.

**Prereqs:** [post-training/reasoning/long-cot-rl.md](./long-cot-rl.md), [post-training/reasoning/length-penalty.md](./length-penalty.md)
**Related:** [post-training/grpo.md](../grpo.md), [post-training/rlvr.md](../rlvr.md)

---

## What it is

A reasoning trace mixes "communication tokens" (must be verbalised because the answer is text) with "computation tokens" (intermediate steps the model would rather think about than write). NF-CoT keeps the first as standard tokens and replaces the second with **continuous thoughts** $\mathbf{z}_t \in \mathbb{R}^d$ drawn from a normalizing flow conditioned on the prefix.

Within one causal stream, two heads alternate:

- text positions → standard LM head → discrete token, cross-entropy loss;
- thought positions → NF head (TARFlow block) → continuous vector, exact log-likelihood under the flow.

The flow is *autoregressive* across thought positions, preserving the left-to-right structure the rest of the model relies on.

## How it works

1. **Distil thoughts from explicit CoT.** Start from a base reasoner trained with explicit CoT. For each example, compress consecutive intermediate tokens into a small number of continuous thoughts using a learned encoder.
2. **Fit the flow.** Train the NF head with maximum likelihood on these distilled thoughts, conditioned on the LLM's hidden state at each thought position. TARFlow gives a tractable density via masked-autoregressive coupling layers.
3. **Joint decode.** At inference, the schedule decides which positions are text and which are thoughts (fixed template or learned router). For each:
   - text: sample a token from the LM head; advance the KV cache normally.
   - thought: sample $\mathbf{z}_t$ from the NF head; inject $\mathbf{z}_t$ as an input embedding at that position; advance KV cache.
4. **Optional RL.** Because the flow yields exact $\log p(\mathbf{z}_t \mid \cdot)$, policy gradients work directly in latent space — no surrogate, no straight-through. This is the key advantage over discrete-latent approaches.

## Why it matters

- **Save tokens without losing tractability.** Existing latent-reasoning methods either lose KV cache (re-attention costs) or lose sampling (vector blending). NF-CoT keeps both.
- **RL in latent space is honest.** Exact likelihoods mean PPO / GRPO style updates apply unchanged; you don't need REINFORCE through a Gumbel-softmax or a learned critic.
- **A drop-in for explicit CoT.** Codebases already optimised around autoregressive decoding (vLLM, SGLang) need minimal surgery: a second output head and a small router for thought positions.
- **Improvements on code.** On code-generation benchmarks NF-CoT raises pass rates over both explicit-CoT and prior latent baselines while cutting intermediate-reasoning cost.

## Gotchas & tricks

- **Distillation target matters.** If the encoder that compresses explicit CoT into thought vectors is weak, the flow learns to fit noise; trace quality flows downstream.
- **Thought schedule is a design knob.** Always-thought is unstable (no anchoring tokens); always-text degrades to standard CoT. Mixed schedules (e.g. 4 thoughts then 1 text checkpoint) tend to win.
- **Numerical stability of the flow.** TARFlow layers can produce extreme log-det values early in training; standard NF tricks (Jacobian clipping, soft-coupling) apply.
- **Inference-time guidance.** Classifier-free guidance over the flow is possible and behaves like temperature on the text side — a useful knob for the explore/exploit tradeoff in latent reasoning.

## Sources

- Paper: *Latent Reasoning with Normalizing Flows (NF-CoT)* — Fu, Yu, Tang, Kang, Qin, Zhang, Gu (UPenn / UCSD / Meta), 2026 — [arXiv:2606.06447](https://arxiv.org/abs/2606.06447).
- Paper: *TARFlow* — for the autoregressive normalizing-flow block used as the NF head.
