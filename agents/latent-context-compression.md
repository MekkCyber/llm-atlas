# Latent Context Compression for Agents (Coca-Cola)
*Depth — replace a tool-output text blob with a few continuous latent tokens that preserve the agent's next-action distribution.*

**TL;DR:** LLM agents normally append raw tool outputs as text, which dominates agent context budgets. **Coca-Cola** trains a compression model that maps each tool observation to $K$ continuous latent tokens inserted into the agent's input-embedding space, with an **Action-Preserving Training** objective that KL-matches the agent's next-action distribution between raw and compressed observations. On four web-search QA benchmarks it cuts agent token consumption by 70–89% and runtime by 24–56% while *improving* task performance.

**Prereqs:** [agents/README](./README.md)
**Related:** [dpo](../post-training/dpo.md), [rejection-sampling](../post-training/rejection-sampling.md)

---

## What it is

Three observations drive the design:

1. Tool outputs (search results, file contents, API responses) are the biggest consumer of agent context and the most redundant — most tokens don't change the next action.
2. Text is a lossy representation of what the agent actually *uses*. The agent only needs enough information to pick the next action.
3. The agent's input embedding space is already a differentiable interface; inserting arbitrary continuous vectors there is natural.

Coca-Cola replaces the text of a tool observation $o$ with a short sequence of latent vectors $z \in \mathbb{R}^{K \times d}$ ($K$ small, same $d$ as the agent's embedding dim), produced by a trained compression model $C$. The agent reads $z$ in place of the tokenized $o$.

## How it works

### Architecture

- Compression model $C_\phi$: a small encoder (LLM or dedicated transformer) that maps observation text to $K$ latent vectors. $K = 4$–$16$ typical.
- Agent $A_\theta$: a pretrained LLM agent (frozen or jointly fine-tuned).
- Interface: the $K$ latent tokens are inserted into the prompt at the position where the tokenized observation would otherwise appear.

### Action-Preserving Training

Loss on a trajectory step with observation $o$ and ground-truth-next-action distribution target:

$$
\mathcal L_{\text{AP}} = \mathrm{KL}\!\left( A_\theta(\cdot \mid \text{prefix}, o) \,\Vert\, A_\theta(\cdot \mid \text{prefix}, C_\phi(o)) \right)
$$

The compression model is trained so that the agent's *next-action logits* under the compressed observation match the full-text ones. Trajectories are collected from a fixed base agent; only $C_\phi$ is updated. Optionally the agent is updated jointly in a second stage.

### Training data

Agent rollouts on web-search QA benchmarks (four reported); each rollout provides one $(o, A(\cdot|o))$ pair per step. No new annotations needed — the labels are the agent's own action distributions on raw text.

## Why it matters

- **Breaks the "pay for every tool token" assumption.** In production agent stacks, tool outputs dominate cost. A fixed 70–89% drop in token consumption is a serious infra win.
- **Task performance *improves* under compression.** The compressed observation strips noise that the raw text would have put into attention. The agent spends its attention budget on the relevant features $C_\phi$ surfaces.
- **Compatible with any pretrained agent.** The compressor is a sidecar; the agent doesn't need to be retrained to benefit.

## Gotchas & tricks

- **Compressor generalization.** $C_\phi$ is trained on the base agent's action distribution; a stronger (or weaker) downstream agent may need its own compressor.
- **Latent tokens aren't inspectable.** You can no longer eyeball the agent's context to debug what it saw. A text summary channel alongside the latents is a cheap compromise during development.
- **Context-poisoning surface is different.** A malicious tool can't inject text into the prompt anymore, but it *can* inject patterns into the latents. The compressor is now part of the agent's trust boundary.
- **$K$ is the real tradeoff.** Small $K$ loses retrievable detail (the agent may need to re-tool-call); large $K$ erodes the token-budget win. Four to sixteen latent tokens per observation seems to be the sweet spot.
- **Not a free lunch for all tools.** Code-execution outputs, long documents and structured data compress differently; $K$ may need to vary by tool type.

## Sources

- Paper: *Tools Need Not Speak in Text: Latent Context Compression for Agents* — Tsinghua AI Agent Group, 2026 — [arXiv:2611.06500](https://arxiv.org/abs/2611.06500).
- Code: [THU-Agent/Coca-cola](https://github.com/THU-Agent/Coca-cola).
- Background: *AutoCompressors* — Chevalier et al., 2023 — earlier work on compressing long context into summary vectors (non-agent setting).
