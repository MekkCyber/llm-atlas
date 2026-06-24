# Trajectory-Augmented Policy Optimization (TAPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO extension that turns the *structure* of wrong rollouts into explicit training trajectories. During RL, sample both correct and incorrect rollouts to the same prompt, then construct **micro-reflective trajectories**: keep the wrong prefix up to the point of failure, splice in a natural-language diagnosis, then a corrected continuation from a sibling correct rollout. Adds difficulty-aware candidate selection and decoupled advantage. Beats GRPO at matched step count on AIME 2024, AIME 2025, HMMT 2025.

**Prereqs:** [../grpo.md](../grpo.md), [../_rl.md](../_rl.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [../rlvr.md](../rlvr.md), [../../evaluation/aime.md](../../evaluation/aime.md)

---

## What it is

Self-distillation methods (and vanilla GRPO) treat the model's wrong rollouts as *negative signal only* — they're penalised in the advantage. They don't extract *what went wrong, where, and how to fix it*.

TAPO turns the wrong-vs-correct *pairing* within a GRPO sampling group into a constructive supervision signal. Each incorrect rollout becomes a new training trajectory whose prefix is the model's own mistake and whose continuation is a model-style correction.

## How it works

For each prompt in the RL batch:

1. **Sample G rollouts** as usual (GRPO).
2. **Split correct vs. incorrect.** Reward signal classifies each.
3. **Construct micro-reflective trajectories** for each incorrect rollout `o_-`:
   - Identify the *failure step* — the earliest position where `o_-` diverges from a sibling correct rollout `o_+`.
   - Keep `o_-` up to (and including) the failure step.
   - Splice in a natural-language **diagnosis** (an in-distribution reflection sentence such as *"Wait, I subtracted instead of added — let me redo this step."*).
   - Splice in the **corrected continuation** sourced from `o_+`.
4. **Difficulty-aware selection.** Only construct micro-reflective trajectories for prompts in the model's capability *boundary* — too easy (model rarely fails) or too hard (model rarely succeeds) prompts get filtered.
5. **Decoupled advantage.** The advantage for the *constructed* trajectory is computed separately from the original sampled trajectories to prevent gradient interference.
6. **Standard PPO-clip update** on the combined set.

Because each constructed trajectory is anchored in the learner's own prefix, the supervision stays close to on-policy — much closer than KL-style distillation against a privileged teacher.

## Why it matters

- **Wrong rollouts become positive signal.** Standard GRPO discards everything about *how* the error happened. TAPO uses it.
- **On-policy explicit correction.** Existing self-distillation methods minimise KL toward a target distribution; TAPO substitutes explicit, position-anchored corrections that don't pull the model off its own distribution.
- **Compatible with verifiable-reward setups.** No reward model needed — the same accuracy / format rewards as RLVR drive the correct/incorrect split.

## Gotchas & tricks

- **The diagnosis text is the lever.** Bad diagnoses (too generic, too long) hurt. The paper's recipe samples diagnoses from the model itself with a small prompt template, keeping them in-distribution.
- **Failure-step alignment matters.** The diff between `o_-` and `o_+` is only well-defined when their early prefixes agree. Heavily-divergent rollouts can't be spliced and are dropped.
- **Decoupled advantage is critical.** Without it, the constructed trajectory's gradient signal contaminates the GRPO advantage for the original rollouts and training destabilises.
- **Difficulty-aware filtering is what makes the method robust.** Always-easy prompts produce no incorrect rollouts; always-hard prompts produce no correct ones. The boundary is where both exist.

## Sources

- Paper: *Learning from Your Own Mistakes: Constructing Learnable Micro-Reflective Trajectories for Self-Distillation* — Huang, Gao, Dong et al., Qwen (Alibaba) / Tsinghua / PKU, 2026 — [arXiv:2606.18844](https://arxiv.org/abs/2606.18844).
- Background: *DeepSeekMath / GRPO* — Shao et al., 2024 — see [../grpo.md](../grpo.md).
