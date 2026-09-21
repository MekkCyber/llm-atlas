# Vision-RL2: Region-Level Policy Optimization for MLLM Perception
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fine-grained perception in multimodal LLMs breaks into two operations — **localizing** the region of interest and **recognizing** its content — and localization tolerates roughly 3–4× stronger token compression than recognition. Vision-RL2 exploits that asymmetry: a small **proposal network** picks candidate regions from a coarse view; the picked region is passed to the MLLM at full resolution for the answer. The proposal network is trained with **region-level RL** where a frozen MLLM reader scores each proposal by how removing it changes the answer likelihood — no annotations, no response sampling, no reasoning trajectories.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [README.md](./README.md) · [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

Standard practice in MLLM perception is to raise input resolution. That inflates the number of visual tokens, which multiplies vision-encoder cost and language-model prefill cost together. The naïve alternative — decoding coordinates directly with the MLLM — is end-to-end trainable but costs a full model pass per query and needs strong grounding ability. A lightweight proposal network distilled from the MLLM's attention is fast but inherits the noise of its attention targets, and — critically — the answer-selection step is a **discrete region choice** whose faithfulness to the answer can't supervise the network.

Vision-RL2's answer: skip the surrogate loss. Optimize the proposal network with **reinforcement learning at the region level**, using the frozen MLLM reader itself as the reward function.

## How it works

**Action space.** Coherent image regions are the actions. The proposal network outputs a distribution over regions (or scores per region).

**Reward.** For each candidate region $R$, run the frozen MLLM reader on the query with and without $R$. The reward is the *change* in answer log-likelihood: $r(R) = \log p(\text{answer} \mid q, I) - \log p(\text{answer} \mid q, I \setminus R)$. Regions whose removal hurts the answer are useful; regions whose removal doesn't matter are noise.

**Objectives.**
- **Subtractive:** penalizes distractor proposals (their removal doesn't change the answer).
- **Additive:** rewards recovery of missing evidence (adding the region increases answer likelihood).

The two run jointly. Only the proposal network updates — the reader stays frozen. No region annotations, no response sampling from the reader, no reasoning traces required.

**Inference-time use.** The trained proposer picks a small set of regions; the MLLM runs on a **sparse encoding** that magnifies those regions and drops background tokens.

## Why it matters

- **Token budget crash.** Vision-RL2 matches or exceeds the base model's largest-budget accuracy with **~4× fewer visual tokens** across six fine-grained benchmarks and four MLLM backbones.
- **Label-free training.** No region annotations, no supervised bounding boxes — the reader's own answer likelihood is the supervision signal.
- **Modular.** Reader stays frozen; only the small proposer updates. Drops into an existing MLLM stack without retraining the language model.
- **Splits the resolution budget correctly.** Coarse view for localization, high resolution for recognition — matches the operations' actual requirements.

## Gotchas & tricks

- **Distractor gradient dominates early.** The subtractive signal is usually noisier than the additive; balancing coefficients matters. Warm up with additive weight, add subtractive as the proposer stabilizes.
- **Region granularity is a hyperparameter.** Too fine → many low-reward regions; too coarse → the "region" is the whole image and the reward collapses. Follow the paper's per-backbone defaults as a starting point.
- **Reader must be strong enough.** If the frozen MLLM can't answer the question well from the ground-truth crop, the reward signal is unreliable. Screen the training set against reader accuracy before RL.
- **Composes with resolution.** Vision-RL2 is orthogonal to raising the recognition-time resolution; the two stack for further gains but the marginal drops fast.
- **Not an alignment method.** This is a perception-efficiency mechanism. It doesn't change what the model believes about the world — only which pixels it looks at.

## Sources

- Paper: *Region-Level Policy Optimization for Fine-grained MLLM Perception* — Shi, Pei, Dong, Xu, University of Sydney, 2026 — arXiv:2609.19745.
