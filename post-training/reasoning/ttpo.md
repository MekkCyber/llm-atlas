# TTPO — Test-Time Policy Optimization
*Depth — RL-style policy updates at inference time, without ground-truth labels.*

**TL;DR:** Standard reasoning post-training (RL, on-policy self-distillation) needs a labeled reward at every prompt. TTPO removes the label by fusing self-distillation with RL asymmetrically: the model teaches itself as it thinks, using its own strong branches to supervise weaker ones. Reported to match label-supervised methods on competition-level math with no ground truth at test time.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [../grpo.md](../grpo.md), [../rejection-sampling.md](../rejection-sampling.md)
**Related:** [../../inference/_test-time-scaling.md](../../inference/_test-time-scaling.md), [../../inference/criticl.md](../../inference/criticl.md), [../entropy-collapse.md](../entropy-collapse.md)

---

## What it is

TTPO — Test-Time Policy Optimization — is a per-prompt update loop that runs during inference. Given one test prompt, generate many completions, extract a self-supervised signal (best-of-branch, self-consistency, or critique from a stronger reasoning branch), and apply a lightweight policy update to the model before emitting the final answer. No labeled reward function is available at test time; the signal is manufactured from the model's own outputs.

## How it works

Per test prompt:

1. Sample `K` reasoning trajectories (roll out at moderate temperature).
2. Build an *asymmetric* supervision signal: strong branches (e.g., ones that self-consistency picks) act as teachers for weaker branches. Distillation loss on weak branches; group-relative RL advantage on strong branches.
3. Apply a small, temporary policy update (adapter-only in practice) that pushes the model toward the strong-branch behavior.
4. Emit the final answer from the updated policy.

The asymmetric design is the key trick: pure self-distillation collapses onto whatever majority mode already exists; pure RL without labels has no signal. Combining them lets the RL half explore while the distillation half stabilizes, so the model can improve on this specific prompt without collapsing.

## Why it matters

Test-time compute has been dominated by "generate more, pick better" (best-of-N, self-consistency, MCTS). TTPO adds a new axis: **spend compute on updating the model** rather than only on sampling from it. If the recipe scales, inference-time gains no longer plateau where verification plateaus.

## Gotchas & tricks

- Applied per prompt — must be cheap. Use a LoRA adapter that is reset between prompts, or a very small learning rate on the last few blocks.
- Requires enough branch diversity to have "strong" vs "weak" branches at all; a model already in [entropy collapse](../entropy-collapse.md) has no distillation signal.
- Not a replacement for offline RL — best composed with an RL/SFT base model; TTPO refines what the base can already partly do.
- Distinct from classical Test-Time Training (TTT) for vision: no augmentation-invariance loss; the objective is fully self-generated.

## Sources

- Paper: *TTPO: Test-Time Policy Optimization* — Wang et al., 2026 — [arXiv:2608.27448](https://arxiv.org/abs/2608.27448)
