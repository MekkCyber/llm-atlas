# PR-OPD — Privileged Representation On-policy Self-Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When an agent gets a skill description, its *hidden states* change far more than its *output probabilities*: in a WebShop study, a skill prompt raised success 42.2 → 56.2 % but shifted <25% of output-token probabilities and >80% of hidden states. PR-OPD exploits this by distilling the privileged-teacher's hidden representations into the student (same model, no skill prompt), reshaping its internal computation while keeping the output head. Beats GRPO on ALFWorld and WebShop with two backbones.

**Prereqs:** [on-policy-distillation](../post-training/on-policy-distillation.md), [grpo](../post-training/grpo.md)
**Related:** [_rl](../post-training/_rl.md), [branch-opd](../post-training/branch-opd.md)

---

## What it is

A self-distillation recipe for agentic RL where the teacher and student are the same model, but the teacher is conditioned on a privileged skill description the student does not see. The transfer target is the teacher's layer-wise *hidden states*, not its output probabilities. "On-policy" because the student rolls out, "self" because the teacher shares weights.

## How it works

1. **Warm-start** the policy with GRPO on verifiable task reward.
2. **Prepare the teacher**: identical weights, conditioned at inference on a text skill description $s$.
3. **Student rollout** without $s$.
4. For each visited state, compute hidden states at a selected set of layers for both teacher (sees $s$) and student (doesn't).
5. **Project** the student's hidden states through a learned linear head to the teacher's hidden-state space (handles basis drift).
6. **Loss**: layer-wise MSE (or cosine) between projected student hidden state and teacher hidden state, summed over tokens.
7. Keep the standard RL reward term for stability; PR-OPD is added, not replaced.

The output head is unchanged — the student learns an *internal* mechanism that mimics acting with the skill, without having to be told the skill at inference.

## Why it matters

A key observational finding before any training: a skill prompt is far more visible inside the model than at its output. Standard RL loses most of that signal because it only pushes on outputs. Hidden-state distillation captures it.

Reported gains over GRPO:

- **ALFWorld:** +4.7 pp success at best.
- **WebShop:** +14.0 pp accuracy at best.
- Best overall across every evaluated setting (two backbones × two environments).

Practical implication: when skills can be expressed as text instructions, PR-OPD transfers them without inference-time prompt overhead and without a dedicated tool-use head.

## Gotchas & tricks

- Projection head matters: without it, the student and teacher hidden spaces drift under the same weights after any training, and MSE goes to zero trivially via that drift.
- Choice of which layers to distill: paper distills across *all* layers; shallower-only recipes underperform.
- Must retain the base RL reward — pure hidden-state matching collapses behavior.
- Only validated for *text-form* privileged skills so far. Non-textual privileges (ground-truth state, API docs embedded as structured data) are an open extension.

## Sources

- Paper: *PR-OPD: Privileged Representation On-policy Self-Distillation for Agentic Reinforcement Learning* — Yang et al., 2026 — [arXiv:2609.36642](https://arxiv.org/abs/2609.36642)
