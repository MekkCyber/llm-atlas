# Evolution Strategies for LLM post-training
*Depth — a memory-efficient, gradient-free alternative to policy-gradient RL for LLM reasoning.*

**TL;DR:** Evolution Strategies (ES) optimize an LLM's parameters by sampling many parameter-space perturbations, evaluating each on a reward, and moving toward the winners. Because updates don't require backprop through rollouts, ES is memory-cheap for reasoning post-training and — as recent analysis shows — preserves output diversity (better Pass@K) where on-policy methods like GRPO collapse to a single mode.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [entropy-collapse.md](entropy-collapse.md), [rejection-sampling.md](rejection-sampling.md), [long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

ES treats the LLM as a black box: perturb the weights `θ + σ·ε_i` for many `ε_i ~ N(0, I)`, score each perturbed model on prompts with a scalar reward, and take a weighted average step in `ε`-space. No backprop through the rollout — just forward passes plus a `θ` update proportional to `Σ r_i · ε_i`. Used since the OpenAI ES paper (2017); recently reframed as a lightweight LLM post-training paradigm.

## How it works

Per iteration:

1. Sample `N` noise vectors `ε_i` in weight-space (usually applied to a low-rank / sparse subset to keep memory manageable).
2. Roll out the perturbed model `π(θ + σε_i)` on a batch of prompts, get scalar rewards `r_i` (task-verifier, GRPO-style outcome reward, etc.).
3. Normalize rewards (rank-based or z-score) → `r̃_i`.
4. Update `θ ← θ + (η / (N·σ)) · Σ r̃_i · ε_i`.

No policy gradient, no KL term, no value model. The "gradient estimate" is a finite-difference-in-parameter-space smoothed by the perturbation scale `σ`.

## Why it matters

- **Memory:** no activations to store for backprop through long rollouts. On a reasoning trace of thousands of tokens this saves 10× or more over PPO/GRPO.
- **Coverage:** the parameter-space noise acts as an implicit diversity regularizer. On-policy RL sharpens `π` toward the highest-reward token sequence and collapses entropy; ES perturbs `θ` symmetrically and keeps solution modes alive. The Aug-2026 analysis shows ES matches GRPO's Pass@1 while dominating on Pass@K.
- **Compose with GRPO:** a hybrid (GRPO for exploitation, ES for exploration) beats either alone. ES's effective updates are sparse in `θ` even when overall drift is large — a sign it's exploring a low-dim useful subspace.

## Gotchas & tricks

- Reward variance is the whole game. Use rank-based normalization or antithetic pairs (`+ε_i`, `−ε_i`) to halve variance.
- Full-parameter `ε` is expensive to store; apply ES to a LoRA adapter or a chosen sparse subset instead of the full weights.
- Choose `σ` carefully — too small = no signal above float noise; too large = perturbed models decohere and rewards become unlearnable.
- The "sparse effective update" observation matters: ES appears to succeed via a few high-leverage directions, not global drift. Under-explored territory.

## Sources

- Paper: *Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO* — Ba et al., 2026 — [arXiv:2608.27351](https://arxiv.org/abs/2608.27351)
- Precedent: *Evolution Strategies as a Scalable Alternative to RL* — Salimans et al., 2017 — [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)
