# S3AC — Stabilized, Selective, Scalable Actor–Critic for LLM reasoning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** S3AC resurrects actor-critic RL for LLM reasoning at frontier scale. The field has mostly moved to critic-free methods (GRPO) because training a value model on long reasoning trajectories is unstable. S3AC bundles three ingredients — a stabilization recipe for value learning, a **selectivity** rule that updates the critic only on prompts where it carries signal, and **DAPS** (Difficulty-Aware Prompt Selection) for scalable curriculum — and reports gains over a strong modern GRPO variant across **1.5B → 397B** parameter scales.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md), [rl-prompt-curation.md](./rl-prompt-curation.md)

---

## What it is

Classical PPO-with-critic for LLMs trained the value head jointly with the policy; the critic was expensive and often unstable, which is why DeepSeekMath replaced it with GRPO's group-mean baseline. S3AC's claim is that the critic is still worth it — the per-token advantage estimate it gives is sharper than the group mean, especially for long trajectories — but only if value learning is stabilized and the critic is updated selectively.

## How it works

Three ingredients:

1. **Stabilization recipe.** Modifications to value-head training that reduce variance on long reasoning rollouts — e.g. return clipping, truncated-bootstrap targets, and a shaped TD objective aligned with the actor's rollout boundaries. The paper gives a concrete recipe.
2. **Selective critic updates.** Not every prompt teaches the critic the same amount. S3AC updates the critic only on prompts where the actor's rollouts span a wide reward spread — those are the prompts whose per-token values are most informative. This saves critic compute and reduces overfit on easy or trivial prompts.
3. **DAPS — difficulty-aware prompt selection.** Keep the training distribution in the hard-but-not-impossible band. Each prompt is scored by recent actor pass-rate; prompts the actor currently gets right or wrong 100% of the time are deprioritized, mid-range prompts are oversampled. This is a cheap, running-average form of curriculum RL.

Together, (1) + (2) make the critic trainable, and (3) ensures the critic has useful signal to learn from.

## Why it matters

GRPO's efficiency win came from giving up on the critic. S3AC says you can have both: tighter credit assignment *and* trainable stability, at least once you add the right stabilization and gate critic updates by prompt informativeness. If the recipe reproduces, it reopens actor-critic as a viable path for long-CoT RL, where credit assignment matters most.

## Gotchas & tricks

- **Three ingredients, not one.** Any of the three alone (stabilization, selectivity, DAPS) is weaker than the bundle. The ablations in the paper are what to look at.
- **Critic size matters.** S3AC's critic is a scalar head on a shared backbone; a from-scratch critic model would re-introduce the compute cost GRPO was escaping.
- **DAPS is a prompt-curation technique.** It can be lifted into GRPO or any critic-free method independently — related to [rl-prompt-curation.md](./rl-prompt-curation.md).

## Sources

- Paper: *S3AC: Stabilized, Selective, and Scalable Actor–Critic for LLM Reasoning* — Cheng, Lyu, Wang, Sun, Zhang, Shanghai AI Laboratory, 2026 — [arXiv:2610.35518](https://arxiv.org/abs/2610.35518).
