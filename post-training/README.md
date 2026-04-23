# Post-Training

*How pretrained models are refined — via supervised fine-tuning, reinforcement learning, preference optimization, and reasoning-elicitation — with the algorithms, reward signals, and rollout infrastructure that turn a base model into something useful.*

---

## What This Is

A pretrained base model predicts next tokens; it doesn't follow instructions, refuse harmful requests, or reason step-by-step on demand. Post-training is the collection of techniques that *shape* behavior without retraining from scratch.

This folder is organized around three axes — **fine-tuning** (supervised adaptation), **reasoning** (techniques that produce step-by-step thinking, mostly RL-based), and **the rest** (general RL, preference optimization, reward design, and pipeline concepts that aren't reasoning-specific).

---

## How to navigate

### 1. Start with the maps

- **[_post-training](_post-training.md)** — the whole post-training landscape: supervised vs preference vs reward-based. Pipeline shape.
- **[_rl](_rl.md)** — the RL side in depth: policy gradient, PPO, GRPO, KL, advantage estimation. Math formulas, intuitive walk-through.
- **[_rewards](_rewards.md)** — where the scalar reward signal comes from: rule-based, ORM, PRM, preference RM, shaping.

### 2. General post-training depth files (root of this folder)

- **[ppo](ppo.md)** — Proximal Policy Optimization. The classical-RLHF ancestor of GRPO; clipped-ratio policy-gradient with a value network.
- **[grpo](grpo.md)** — Group Relative Policy Optimization. The policy optimizer behind most modern reasoning RL.
- **[dpo](dpo.md)** — Direct Preference Optimization. The closed-form, rollout-free alternative to PPO-RLHF for pairwise preference data.
- **[rlvr](rlvr.md)** — RL with Verifiable Rewards. Rule-based reward signal, no learned RM.
- **[rejection-sampling](rejection-sampling.md)** — generate K candidates, filter, SFT on the survivors. Used heavily in Llama 3 and R1.
- **[cot-reward-model](cot-reward-model.md)** — a learned RM that *generates* a reasoning trace + JSON judgment; 84% → 98% verifier-accuracy gap over value-head RMs (Kimi k1.5).
- **[rl-prompt-curation](rl-prompt-curation.md)** — the four data-curation rules for RL prompt pools, from Kimi k1.5. Includes the "easy-to-hack" filter.

### 3. Subfolders

- **[fine-tuning/](fine-tuning/)** — supervised adaptation: SFT, LoRA, adapters, merging. The signal is "imitate this reference completion."
- **[reasoning/](reasoning/)** — elicit step-by-step thinking: long-CoT RL, PRMs, ORMs, MCTS. The signal is "did the solution verify?"

---

## Reading Order

A single coherent path through the folder, once the maps are read:

1. [_post-training](_post-training.md) — the overall map.
2. [fine-tuning/](fine-tuning/) — start with SFT basics (the common ancestor of every recipe).
3. [_rl](_rl.md) — policy-gradient fundamentals before jumping into specific algorithms.
4. [_rewards](_rewards.md) — the five reward families.
5. [ppo](ppo.md) — the classical baseline; foundation for GRPO and mirror descent.
6. [rlvr](rlvr.md) and [grpo](grpo.md) — the modern-default reasoning-RL stack.
7. [dpo](dpo.md) — offline preference optimization without rollouts.
8. [rejection-sampling](rejection-sampling.md) — how production recipes generate SFT data from RL checkpoints.
9. [rl-prompt-curation](rl-prompt-curation.md) — RL-prompt-pool data engineering.
10. [cot-reward-model](cot-reward-model.md) — generative verifiers for RLVR.
11. [reasoning/](reasoning/) — long-CoT RL, mirror descent, length penalty, long2short, PRM/ORM, MCTS.

---

## Related

- [pre-training/](../pre-training/) — where the base model comes from.
- [systems/](../systems/) — the rollout and orchestration infrastructure post-training depends on.
- [evaluation/](../evaluation/) — reward models and reasoning models are evaluated here.
- [safety/](../safety/) — safety RL and refusal training.
- [case-studies/](../case-studies/) — end-to-end post-training pipelines (DeepSeek-V3, DeepSeek-R1, Kimi k1.5, OLMo 2).
