# Long-Context RL
*Depth — execution-stack techniques for RL post-training at million-token sequence lengths.*

**TL;DR:** Modern LLM inference reaches million-token contexts, but RL post-training typically caps at 128K–256K and relies on length generalization to bridge the gap. Long-context RL brings the two into alignment by rearranging rollout scheduling, activation memory, and gradient accumulation so a fixed GPU budget can train on 2M+ token trajectories. Instantiated on GRPO in the LongStraw stack.

**Prereqs:** [partial-rollouts](partial-rollouts.md), [../post-training/grpo](../post-training/grpo.md), [dualpipe](dualpipe.md)
**Related:** [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md), [../post-training/rlvr](../post-training/rlvr.md)

---

## What it is

RL post-training pipelines that keep sequence length at the same order of magnitude as deployment inference (roughly 10⁶ tokens) rather than the traditional 10⁵. Aimed at agentic tasks where tool outputs, observations, and prior decisions accumulate over long trajectories, so training-time length generalization is not enough.

The subject is *systems*, not the RL algorithm: the algorithm is usually GRPO or a close variant, and the depth-file focus is on the execution-stack tricks that make million-token rollouts and gradient steps fit in a fixed GPU budget.

## How it works

Three axes of pressure at million-token scale:

1. **Rollout collection.** A group of `G` rollouts (GRPO needs a group for the baseline) at 2M tokens each is `G × 2M` sequence positions to generate. Partial-rollout schedulers pipeline generation with training so GPUs are never idle between epochs.
2. **Activation memory.** Naive backprop-through-time on 2M-token sequences exceeds any per-GPU HBM. Techniques: (a) sequence-parallel gradients (shard sequence dim across devices), (b) activation offload to CPU or NVMe with prefetch, (c) selective recomputation of attention activations.
3. **Gradient stability.** With group size `G` at 2M tokens, per-group variance of the GRPO baseline grows. Practical stacks widen the group only modestly and lean on trajectory-level normalization or reward shaping to keep advantages well-scaled.

The LongStraw stack (2026) is architecture-aware: it inspects the target model's attention pattern (dense vs sliding-window vs MLA-style compressed KV) and picks activation partitions per layer accordingly, since the memory profile differs by attention variant.

## Why it matters

- **Closes the deployment/training gap.** Agents deployed on million-token contexts benefit from being *trained* there, not just generalized there.
- **Unblocks long-horizon agent RL.** Tool-using agents accumulate long observations; RL post-training at 256K forced pipelines to summarize or truncate, which the RL signal then had to learn to work around.
- **Fixed GPU budget claim.** LongStraw's contribution is not "throw more GPUs at it" but rearranging the same budget — the interesting scaling result is that partial-rollout scheduling plus per-layer activation partitioning yields the extra order of magnitude for free.

## Gotchas & tricks

- **Reward sparsity gets worse.** A 2M-token trajectory with a single outcome reward is a much sparser learning signal per token than at 256K. Pair with process rewards or on-policy distillation to densify.
- **KV cache management dominates.** For inference-style rollouts embedded in the training loop, paged KV or MLA-style compressed KV can be the difference between fitting and OOM.
- **Attention pattern matters.** Sliding-window or MLA attentions have very different activation profiles from dense causal attention; the execution planner has to know.
- **Length generalization is still needed.** Even at 2M-token training, deployment can push further. Long-context RL raises the training ceiling but does not replace test-time length robustness (RoPE scaling, DCA, etc.).

## Sources

- Paper: *LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget* — Zhou et al., 2026 (Mind Lab) — the anchor system, instantiated on GRPO.
- Related: [partial-rollouts.md](partial-rollouts.md), [../post-training/grpo.md](../post-training/grpo.md).
