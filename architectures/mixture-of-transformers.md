# Mixture-of-Transformers (MoT)
*Depth — a modality-/role-routed split where different token classes flow through different transformer weights but share attention layers.*

**TL;DR:** Ordinary transformers apply the same MLP and attention weights to every token, regardless of its role or modality. **Mixture-of-Transformers** replaces the shared parameters with *modality-routed* ones: text tokens, visual tokens, and (in world-model settings) world-state tokens each pass through their own dedicated weight matrices in the block's MLP and often in QKV projections, while attention itself still lets all token classes mix. Unlike token-level MoE, routing is by *token role*, not by a learned router — the choice is deterministic and cheap.

**Prereqs:** [transformer-block](transformer-block.md), [_moe](_moe.md)
**Related:** [../multimodal/world-state-registers](../multimodal/world-state-registers.md)

---

## What it is

For a transformer block with parameters $\theta$, MoT partitions $\theta$ into per-role subsets $\{\theta_r\}_{r \in \mathcal{R}}$ where $\mathcal{R}$ is a small fixed set of token classes (e.g. `{visual, world_state}` or `{text, image, audio}`). During the block's MLP (and typically QKV projection), each token uses only the weights for its role. The attention operator remains shared, so cross-role interactions still happen — MoT does not partition information, only *processing capacity*.

## How it works

Given input $x = [x^{(1)}, \dots, x^{(N)}]$ with per-token roles $r(1), \dots, r(N)$:

1. Compute per-role QKV: $Q_i, K_i, V_i = W_{q,r(i)} x_i, W_{k,r(i)} x_i, W_{v,r(i)} x_i$.
2. Run standard multi-head attention across all tokens together (no partitioning).
3. Apply per-role MLP: $\text{MLP}_{r(i)}(z_i)$.
4. Add residual, layer-norm — usually shared across roles.

Because the router is deterministic (based on token type, not learned), there's no load-balancing loss, no capacity factor, no auxiliary MoE machinery. The block roughly doubles parameter count for two roles, but activation cost per token stays the same as a single-role block of the same width.

## Why it matters

Text and pixel tokens have very different statistics; forcing shared MLPs into service for both means each acts as a compromise. In world-model architectures the gap is even wider — world-state registers are dense, low-count, and highly-updated, while visual tokens are numerous and locally-structured. MoT lets each token class have exactly-tuned processing capacity while keeping cross-token interactions in a shared attention layer. Reported to be the piece that makes register-based multi-agent video diffusion trainable at scale.

## Gotchas & tricks

- **Doesn't replace token-level MoE.** MoT partitions by role; sparse-MoE partitions by learned routing within a role. They compose.
- **Attention must stay shared.** Splitting attention weights per role breaks the cross-role information flow that made MoT worth doing.
- **Norm layers are subtle.** Per-role LayerNorm sometimes hurts because it decouples the gradient scales; shared norm is the safer default unless per-role norm demonstrably helps.
- **Doubles memory, not compute.** Per-role weights all live in memory; only the active role's weights are used per token during forward.

## Sources

- Paper: *Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers* — Mo, Li, Leng, Singh, Zhou (UCLA & Adobe), 2026 — [arXiv:2607.21594](https://arxiv.org/abs/2607.21594).
- Earlier related pattern: modality-specific MLPs in vision-language transformers (used in various VLM stacks; MoT formalizes and names the pattern).
