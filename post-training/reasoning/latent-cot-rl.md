# Latent CoT with On-Policy RL (Switchable Boundary Tokens)
*Depth — training latent (hidden-state) chain-of-thought with standard GRPO by anchoring the latent block with explicit boundary tokens.*

**TL;DR:** Latent chain-of-thought (Coconut-style) replaces visible reasoning tokens with continuous hidden-state recurrence — cheap and dense, but hard to optimize with on-policy RL because there's no token-level action surface inside the latent block. **Switch** (2026) adds two boundary tokens `<swi>` and `</swi>` that mark the latent region; the policy chooses *when* to enter and exit latent mode at the token level, GRPO updates on those boundary actions, and the latent block becomes trainable and inspectable. Reaches 79.3% on MATH-500.

**Prereqs:** [../grpo.md](../grpo.md), [long-cot-rl.md](long-cot-rl.md)
**Related:** [../_rl.md](../_rl.md)

---

## What it is

A modification of latent-CoT inference and training that makes the latent block:

- **RL-trainable** with standard GRPO (no specialized policy-gradient machinery).
- **Mechanistically inspectable** — the boundary tokens give a fixed entry/exit point that probes can target.

The model emits visible tokens by default. When it emits `<swi>`, the next K steps run in latent mode (no token sampling; hidden state recurs). On `</swi>`, the model resumes visible decoding.

---

## How it works

### Boundary tokens

Two reserved tokens in the vocabulary: `<swi>` (enter latent) and `</swi>` (exit). The policy decides at every step whether to emit them — this is the trainable action surface for the latent block as a whole.

### Latent recurrence

Inside the latent region, the model performs K recurrent forward passes with no token sampling — the hidden state evolves continuously. At exit, it resumes visible decoding from the final hidden state.

### Why GRPO now works

Without boundary tokens, the entire latent block contributes one diffuse reward signal that can't be assigned to actions. With boundary tokens, the entry decision *is* an action — GRPO can credit-assign over it. The internal recurrence is treated as a deterministic function of `<swi>`'s position, so no gradient flows through latent sampling.

### Mechanistic finding

Switch's analysis localizes the latent computation to a single hidden-state transition at the moment of entry — most of the reasoning collapse happens in one step. This is a concrete interpretability claim, not a vague "latent reasoning works" claim.

---

## Why it matters

- **Bridges latent-CoT and visible-CoT.** Get the compute savings of latent reasoning while keeping RL trainable and behavior inspectable.
- **The boundary-token trick is general.** Any mechanism that wants to mix latent compute with on-policy RL can use the same anchoring trick.
- **79.3% MATH-500.** Competitive with visible-CoT RL methods at lower decode cost.

---

## Gotchas & tricks

- **Latent step count K is a hyperparameter.** Too small: not enough compute to do useful reasoning. Too large: long latent regions destabilize training.
- **Boundary-token frequency control.** Models sometimes learn to over-emit `<swi>` and use the latent block as a hiding place from the reward. Penalize excessive latent time or cap K explicitly.
- **Probes target the boundary.** Mechanistic analysis is most informative around the `<swi>` step; instrument early.
- **Doesn't compose with all decoders.** Speculative decoding interacts oddly with latent regions because the draft model doesn't know to stop sampling; either disable speculative inside latent or train the draft with the same boundary tokens.

---

## Sources

- Paper: *Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning* — Guo et al., HKUST-GZ + Cambridge + NTU, 2026 — [arXiv:2606.13106](https://arxiv.org/abs/2606.13106).
