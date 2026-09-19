# Observation Supervision in Agent SFT (ActObs)
*Depth — a one-line change to agent-trajectory SFT that flips its downstream RL performance.*

**TL;DR:** Standard agent SFT applies the loss only to action tokens and *masks* environment observations as context. **ActObs** removes that mask: observation tokens (already in the trajectory) also get supervised. No new data, params, sequence tokens, or forward passes. Post-SFT the two initializations look similar; post-GRPO ActObs dominates on Terminal-Bench 2.0 and transfers to unseen cross-domain code editing. A gradient-flow analysis explains why: action-only training leaves a large residual observation gradient that degrades environment prediction below the base model, poisoning the RL init.

**Prereqs:** [../grpo](../grpo.md), [../rlvr](../rlvr.md)
**Related:** [../reasoning/long-cot-rl](../reasoning/long-cot-rl.md), [../rejection-sampling](../rejection-sampling.md), [../../evaluation/swe-bench.md](../../evaluation/swe-bench.md)

---

## What it is

Agent trajectories look like:

```
[obs_0] [action_0] [obs_1] [action_1] ... [obs_T] [action_T]
```

**Standard SFT** loss: cross-entropy on action tokens only; observations are context (attended to, not predicted).

**ActObs SFT**: cross-entropy on *both* action and observation tokens. The policy is trained to predict what the environment will emit next as well as what it will do. Because observations are already in the sequence, this adds *zero* extra data, tokens, or compute — just an unmasked loss.

Deployed agents never *generate* observations; the training objective is a proxy for "the model can predict consequences of its actions."

---

## How it works

Set the loss mask for observation tokens to 1 (was 0) during SFT. Everything else — data, sequence, forward pass, hyperparams — stays fixed.

Result at SFT time: on Qwen3-4B/8B, ActObs and action-only produce similar quality checkpoints — the change is invisible in aggregate SFT metrics.

Downstream RL (GRPO on verifiable coding tasks):

- **Qwen3-4B:** ActObs → higher Pass@k at every evaluated sampling budget on Terminal-Bench 2.0.
- **Qwen3-8B:** +3.4 pp Pass@16, more distinct tasks solved (trades some Pass@1 reliability).
- **Cross-domain (aider-polyglot, unseen during SFT+RL):** +4.2 pp Pass@1 at 4B.

The paper's gradient-flow analysis: in action-only SFT, action and observation gradients start out orthogonal, but the observation gradient never shrinks — it stays large and degrades the base model's environment prediction. Joint supervision keeps observation prediction intact, giving RL a healthier initialization to explore from.

---

## Why it matters

- **Free lunch at SFT time, big win at RL time.** Zero-cost training-side change, meaningful downstream gains.
- **Explains a puzzle.** Why does SFT + RL sometimes underperform pure RL? Part of the answer is that action-only SFT actively degrades environment prediction. Fixing the mask fixes the init.
- **Preserves entropy and stays close to init.** ActObs-trained policies retain more entropy during RL and move less from the SFT init — both correlate with better long-horizon exploration.

---

## Gotchas & tricks

- **Only for trajectory data.** ActObs assumes observations are already token-level in the sequence. Not applicable to non-trajectory instruction data.
- **Tokenizer must handle observation content.** Environment output (bash stdout, tool responses) often contains rare tokens that inflate loss noise. Cap observation loss weight or clip if unstable.
- **Doesn't help pure-chat models.** The effect is downstream-of-RL; models never RL'd don't benefit meaningfully.
- **Pass@1 vs. Pass@k tradeoff at large scale.** On 8B, ActObs trades some Pass@1 reliability for higher Pass@k — worth knowing if you deploy at Pass@1.
- **Interacts with observation length.** Very long observations pump up total loss; consider length-normalizing observation loss to keep action loss dominant.

---

## Sources

- Paper: *Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL* — Zhang et al., 2026 (AWS) — [arXiv:2609.20715](https://arxiv.org/abs/2609.20715).
