# Tool-Use RL Collapse

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A diagnosis of a recurring failure mode in agentic multi-step tool-use RL: training looks fine, then performance **abruptly collapses** as tool-call structure breaks. The underlying tool-use capability survives — what fails is the probability of a few **control tokens** (tool tags, separators, schema markers). Interleaving SFT with RL is the most reliable known fix, at a measured cost in OOD-format generalization.

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md), [_rewards](_rewards.md)
**Related:** [on-policy-distillation](on-policy-distillation.md), [rejection-sampling](rejection-sampling.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

In multi-step tool-use RL on top of an instruction-tuned LLM, late-training collapse looks like:

- success rate trends up for many steps,
- the loss and KL look healthy,
- then the success rate **falls off a cliff**, sometimes within a hundred update steps,
- inspection of rollouts shows the model producing malformed tool calls — wrong tag, missing argument, broken JSON.

It's not the *capability* that collapses. If you inspect the policy's distribution over the action structure with the format constraint removed, the model still "knows" the right tool and arguments. What broke is the small set of **structural control tokens** governing tool-call emission.

## How it works

The mechanism the paper isolates:

1. **Control tokens are short** — sometimes one or two tokens — and they sit at the top of a heavily-skewed marginal distribution. They are the highest-probability emission at a few specific positions in the rollout.
2. RL updates against a sparse outcome reward are noisy at the token level. With group-relative advantage (GRPO) over a few rollouts, a single high-leverage token can receive a large normalized advantage.
3. Through several update steps, these control tokens accumulate **unexpected probability spikes** (or dips) — either being slammed to near-1 (over-fires) or near-0 (under-fires). Either is fatal: it breaks the parser the rollout environment uses to detect tool calls, the next step receives no tool output, and the trajectory's reward collapses.
4. Because the failure is *structural*, the reward then collapses uniformly across the batch — there's no gradient signal pushing the model out of the trap.

## Why it matters

- **Mechanistic explanation** for an otherwise mysterious bug: "agentic RL is unstable" hand-waves the gradient story; the control-token diagnosis turns it into a concrete failure mode you can monitor for.
- **Reframes the SFT-in-the-loop tactic** that production teams already use. The paper shows it's not just an inductive bias hack — it actively re-anchors the structural-token marginals to a safe distribution that RL drifted away from.
- **Quantifies an underappreciated cost.** Interleaved SFT+RL solves the collapse but **degrades OOD format-and-content generalization** — once SFT is in the loop, the model commits harder to the training-time tool-call format and recovers more slowly from in-the-wild format variants.

The fix, in practice:

| Recipe | Effect on collapse | Cost |
| --- | --- | --- |
| RL-only | Frequent collapse | None — but unusable |
| Off-policy SFT supervision (frozen) | Slows collapse | Distribution-mismatched |
| Hint-based guidance (inject hints into prompt) | Mild help | Couples to environment |
| Erroneous-example supervision | Mild help | Easy to overfit to fake errors |
| **Interleaved SFT + RL** | **Largest reduction** | **OOD format/content degradation** |

## Gotchas & tricks

- **Watch the control-token marginals during training**, not just the loss. A few hundred steps of monitoring is enough to predict collapse.
- **Learning rate matters more than expected** — high LR accelerates the spike-formation; tighter clipping in the PPO ratio also helps.
- **Shaping rewards make this worse.** Any reward that scores the *format* (e.g., +1 for a parseable tool call) introduces extra incentive to over-tighten control-token marginals. Make format rewards small if you use them at all.
- **OOD eval is the regression to look for.** SFT-interleave will improve in-distribution stability *and* hide a real generalization regression — measure both.
- **General phenomenon.** Although the paper studies tool use, the same mechanism plausibly applies to any agentic RL with high-leverage structural tokens (XML/JSON schemas, function-call markers, MCP tag formats).

## Sources

- Paper: *Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It* — Hao, Jin, Liao, Liu, Zhao, 2026 — [arXiv:2606.26027](https://arxiv.org/abs/2606.26027) — CASIA / UCAS.
- Background: *DeepSeekMath / GRPO* — Shao et al., 2024 — the outcome-RL backbone where this failure mode appears.
- Background: *Tülu 3* — AI2, 2024 — popularized SFT-then-RL pipelines whose stability properties this paper sharpens.
