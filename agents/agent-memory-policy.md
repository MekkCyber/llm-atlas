# Agent Memory Policy (Belief-Entropy RL)
*Depth — RL training of the recursive-summary "memory" policy that long-horizon agents rely on.*

**TL;DR:** Long-horizon LLM agents survive million-token tasks by recursively distilling history into condensed memories. If you train that summarization policy with only the final task reward, errors accumulate silently for hundreds of steps before the outcome reveals them. MMPO (Liu et al., 2026) replaces the sparse outcome signal with a **per-summary Belief Entropy reward**: the model's posterior uncertainty over the latent task state given the current memory. Penalize summaries that spike that entropy and the agent sustains **97.1% of short-horizon performance at 1.75M-token contexts**.

**Prereqs:** [grpo](../post-training/grpo.md), [_rewards](../post-training/_rewards.md)
**Related:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md), [rlvr](../post-training/rlvr.md)

---

## What it is

A reinforcement-learning recipe whose *policy* is the agent's memory-summarization step (the function that turns history → next memory), and whose *reward* is dense and self-supervised: the model's own confidence about the latent task state after applying the candidate summary.

## How it works

The setup:

- Agent runs over a long horizon, periodically summarizing past trajectory into a fixed-size memory $m_t$.
- For each summary action, MMPO defines:

$$
\text{BeliefEntropy}(m_t) = H\big( p_\theta(s_t^\star \mid m_t) \big)
$$

where $s_t^\star$ is a latent task-state variable. The model itself produces the posterior $p_\theta(s_t^\star \mid m_t)$ — no external labels.

- **Dense reward shape.** Each summarization step gets a reward $r_t = - \Delta H_t$ — punish entropy spikes. Combined with terminal task reward for the outer GRPO objective.
- **What the entropy spike means.** A good summary preserves the information that disambiguates the task; a bad summary either drops it or buries it under noise. The first case shows up as entropy *increasing* relative to a longer/cleaner alternative; MMPO directly trains the policy away from those.

The outer loop is standard GRPO over rollouts of length $T$, with the per-step belief-entropy reward summed alongside the terminal outcome reward.

## Why it matters

- **Outcome RL on long horizons is broken.** If the only signal arrives 100 steps after the responsible action, credit assignment collapses. MMPO is one of the cleanest ways to introduce *intermediate*, *self-supervised* supervision without external labels.
- **Sustains long contexts.** At 1.75M-token context lengths MMPO retains 97.1% of short-horizon performance — most other long-horizon agent recipes drop sharply past 100k tokens.
- **Belief entropy is a portable primitive.** Independent of MMPO, the metric (model's posterior uncertainty over a task variable) plugs into any agent setting where you can name a latent task state.

## Gotchas & tricks

- **Naming the latent task state.** $s_t^\star$ must be something the model can plausibly estimate. For coding agents it's "intended program behavior"; for search agents it's "target answer set." Define it explicitly, otherwise the entropy is over unstructured noise.
- **Reward shaping risk.** A reward that *only* asks for low entropy can be gamed by collapsing to overconfident wrong beliefs. Keep the terminal outcome reward in the mix and tune the dense-reward weight low.
- **Compute cost.** Computing the posterior for every summarization step doubles the per-step forward cost. Sub-sample (e.g. every k-th step) for cheap training.
- **Pair with summary length control.** Belief entropy alone doesn't bound summary length; combine with an explicit length budget per memory cell.

## Sources

- Paper: *Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents* — Liu et al., USTC / Zhejiang / Tencent, 2026 — [arXiv:2605.30159](https://arxiv.org/abs/2605.30159) — introduces Belief Entropy and MMPO.
