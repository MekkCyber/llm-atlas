# Multiscale Advantage Estimation (MAE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For **long-horizon agent RL**, align credit assignment with the actual decision structure of the task. Standard RL for LLM agents computes advantages per token, but each assistant *turn* is one meaningful decision — tokens inside a turn are tightly correlated and shouldn't be treated as independent actions. MAE computes **turn-scale advantages** (grouping tokens by turn, applying a group-relative baseline at the turn level) and pairs them with **turn-level importance sampling** so updates and advantages agree. Introduced in Multiscale Policy Optimization (MPO; Ren et al. 2026); shows consistent gains over GRPO on search, tool-use, and automation tasks at 4B and 30B-A3B scale.

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md)
**Related:** [grpo.md](grpo.md) · [_rl.md](_rl.md) · [rlvr.md](rlvr.md)

---

## What it is

In agentic RL, an assistant produces a *turn* (a full tool call, a search query, or a reply) made of many tokens, waits for an environment response, and produces the next turn. Credit for the final reward belongs to the sequence of turn-level decisions. GRPO-style advantages treat each token as a separate decision — but critic values across tokens *within one turn* are tightly correlated: the decision is really made once, at the start of the turn.

MAE accepts this and estimates advantages at the **turn scale**:

- Group tokens into turns.
- Compute a group-relative baseline across *rollouts of the same prompt* at the turn level.
- Each turn gets one advantage; all tokens inside the turn share it.

A matching **turn-level importance sampling** correction ensures the on-policy / off-policy correction operates at the same scale as the advantage estimate.

## How it works

### Turn boundaries and advantages

For prompt $q$, sample $G$ full rollouts. Within each rollout, split tokens into turns $u_1, u_2, \ldots$ (demarcated by environment responses or role switches). Let $R_g$ be the terminal reward of rollout $g$, and $V_{u,g}$ a turn-level critic value (or 0 if critic-free).

Turn-scale advantage for turn $u$ of rollout $g$:

$$
A_{u,g} = (R_g - V_{u,g}) - \frac{1}{G} \sum_{g'} (R_{g'} - V_{u, g'})
$$

(The group-relative baseline is computed per *turn position* across rollouts of the same prompt, akin to GRPO's rollout-level baseline but one scale finer.)

Each token in turn $u$ of rollout $g$ inherits $A_{u,g}$.

### Turn-level importance sampling

Standard token-level IS computes $\pi_\theta / \pi_{\theta_{\text{old}}}$ per token. MPO's turn-level IS aggregates the ratio over the tokens of a turn before applying it, so the correction scale matches the advantage scale. The paper argues this prevents the clipped-ratio surrogate from fighting the turn-coherent credit signal.

### Full loss

PPO-clipped loss on top of the turn-level IS ratio and turn-scale advantages, otherwise analogous to GRPO.

## Why it matters

- **Credit matches decision structure.** The policy update stops punishing or rewarding tokens that were determined by an earlier turn-level decision.
- **Consistent gains across tasks.** MPO reports improvements over strong RL baselines on search, tool use, and automation at both **4B** and **30B-A3B** scale — a sign the fix isn't scale-specific.
- **Pairs naturally with [rejection-sampling.md](rejection-sampling.md).** Turn-level filtering is a sibling idea at a different lever.
- **Minimal implementation cost.** It's a boundary-detection + aggregation layer on top of a GRPO implementation.

## Gotchas & tricks

- **Boundary detection is load-bearing.** If your tokenizer puts tool-call delimiters, role headers, or environment tokens in odd places, mis-grouping tokens undermines everything. Validate on a trace.
- **Variable turn lengths.** Group-relative baselines across rollouts need alignment of turn *positions*, not just token positions. The paper uses turn index $u$; rollouts with different numbers of turns need a well-defined protocol (truncate, pad, or weight).
- **Not a replacement for GRPO on single-turn tasks.** On math/code with one assistant turn, MAE reduces to rollout-level GRPO. Use the simpler one where it applies.
- **Still outcome-only.** MAE improves credit *alignment* but doesn't introduce per-turn rewards. For tasks where intermediate steps should get specific feedback, pair with a process reward model.
- **Interacts with clip epsilon.** The turn-level IS ratio has higher variance than token-level; aggressive $\epsilon$ can clip real signal. The paper tunes this per task.

## Sources

- Paper: *Multiscale Policy Optimization for Long-Horizon LLM Agents* — Ren et al., 2026 — [arXiv:2610.35013](https://arxiv.org/abs/2610.35013)
- Related: *DeepSeekMath (GRPO)* — Shao et al., 2024 — the rollout-scale baseline MAE refines.
