# Entropy collapse in on-policy RL post-training
*Depth — the standing failure mode of on-policy RL for LLMs and what causes it.*

**TL;DR:** Under on-policy RL (PPO, GRPO, RLVR variants), the policy sharpens toward the highest-reward completions and its per-token entropy drops fast. Pass@1 climbs but Pass@K flattens or falls: the model can now solve the easy prompts more reliably but has lost the diversity needed for the hard ones. Understanding entropy collapse is the entry point for most modern RL-post-training tricks (KL penalties, ES hybrids, temperature scheduling, long-cot regularization).

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md), [ppo.md](ppo.md)
**Related:** [evolution-strategies.md](evolution-strategies.md), [long-cot-rl.md](reasoning/long-cot-rl.md), [rlvr.md](rlvr.md)

---

## What it is

An on-policy RL update raises the probability of high-reward sequences and lowers everything else. Over many steps, the policy's output distribution over completions concentrates on one "template" — same reasoning skeleton, same phrasing, same wrong answer when it fails. Measured either as a drop in per-token entropy `H(π(·|s))` or as `Pass@K` degrading toward `Pass@1`.

## How it works

Two mechanisms combine:

1. **Reward sharpening.** The gradient `∇ log π(a|s) · A(s,a)` pushes probability mass onto exactly the completions that got positive advantage in the current batch. With no counter-force, iterating this converges to a delta distribution.
2. **Sampling bias.** On-policy rollouts under a low-entropy policy no longer sample the outcomes that would give useful gradient. The model can't learn from behaviors it never emits, so exploration self-terminates.

Standard mitigations:

- **KL penalty to a reference model** (`β · KL(π ‖ π_ref)`) — the canonical PPO/RLHF term. Slows collapse but doesn't stop it.
- **Entropy bonus** in the loss (`-λ · H(π)`). Small `λ`; larger values destabilize.
- **Temperature schedule** at rollout time. Cheap but tricky to tune.
- **Diversity-preserving optimizers** (ES, hybrid GRPO+ES) that perturb in parameter-space instead of trajectory-space.
- **Group-relative advantage** (GRPO) helps by removing constant biases but does not by itself prevent collapse.

## Why it matters

Every current reasoning-RL recipe fights this. It's the reason Pass@K is now the default reasoning metric alongside Pass@1: high Pass@1 with dead Pass@K signals a model that can no longer be extended by inference-time compute (best-of-N, majority voting, search). A collapsed policy also transfers badly — SFT + on-policy RL to green a benchmark, then discover the model has narrowed on that benchmark's phrasing.

## Gotchas & tricks

- Track `H(π)` per step, not just reward — reward can improve for a while after entropy is already gone.
- KL to reference helps most when reference is diverse; a badly-tuned SFT reference doesn't save you.
- Long-CoT training is a natural entropy sink because chains are longer and each token multiplies constraints — expect faster collapse and budget more mitigations.
- Rewarding only outcome (RLVR) collapses faster than rewarding process (PRM) because the shaping signal is coarser.

## Sources

- Empirical analysis: *Understanding Evolution Strategies for LLM Reasoning* — Ba et al., 2026 — [arXiv:2608.27351](https://arxiv.org/abs/2608.27351)
- Long-CoT context: see [long-cot-rl.md](reasoning/long-cot-rl.md).
- Original PPO KL term: Schulman et al., 2017.
