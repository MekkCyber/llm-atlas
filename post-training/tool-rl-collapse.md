# Tool-Use RL Collapse
*Depth — a reproducible failure mode in agentic RL and the supervisory-signal menu that fixes it.*

**TL;DR:** Multi-step tool-use RL on language models often **collapses catastrophically**: task performance abruptly drops to zero and tool-invocation structures fail. The root cause is not capability loss — the underlying tool-use ability is intact. Instead, RL drives **probability spikes on a small set of control tokens** (function-call delimiters, JSON brackets, tool-name tokens) that destroy the model's ability to emit valid structured outputs. Interleaving SFT with RL stabilizes training but trades against OOD generalization; off-policy supervision and erroneous-example supervision sit at different points on the same frontier.

**Prereqs:** [grpo.md](grpo.md), [_post-training.md](_post-training.md), [_rl.md](_rl.md)
**Related:** [opid.md](opid.md), [rejection-sampling.md](rejection-sampling.md), [rl-prompt-curation.md](rl-prompt-curation.md)

---

## What it is

A specific, reproducible failure mode observed when training language agents with outcome-only RL (GRPO/PPO) on multi-step tool-use tasks. Symptoms:

- Performance climbs early, then **abruptly drops to ~0** over a small number of steps.
- Generated outputs look syntactically broken — malformed function calls, missing brackets, mangled tool names.
- A pre-collapse checkpoint, when prompted differently, still demonstrates correct tool-use capability — so it's a *format* failure, not a capability loss.

Diagnostic: probability mass on a few control tokens has been pushed near 1, destroying the structured-output distribution that downstream parsers depend on.

## How it works

The mechanism is straightforward once you look at the token-level distribution:

- Outcome RL rewards trajectories that complete the task. Many trajectories begin with the same control tokens (e.g., the `<tool_call>` delimiter).
- The policy update increases probability on those high-frequency, high-utility control tokens.
- Without a counter-pressure, the policy converges to **near-deterministic emission** of control tokens regardless of context — wrecking the parsing pipeline and collapsing task success.

The paper's mitigation menu, with documented tradeoffs:

| Supervisory signal | Stability | OOD format | OOD content |
| --- | --- | --- | --- |
| **Interleaved SFT + RL** | Best stability | Degrades | Degrades |
| Off-policy supervision | Moderate | Better than interleaved | Moderate |
| Hint-based guidance | Moderate | Moderate | Best |
| Erroneous-example supervision | Moderate | Best | Moderate |

The interleaving scheme is the simplest fix; the others trade some stability for better OOD generalization on either format or content.

## Why it matters

- Names a failure mode that has been hitting agent-RL teams without a shared vocabulary. "My tool calls broke" is now a category, not a mystery.
- Gives a documented menu of supervisory signals with tradeoffs, short-circuiting trial-and-error.
- Reinforces a structural lesson: **outcome reward alone is not a complete supervision signal for agents** — you need something that protects the format-control subspace.

## Gotchas & tricks

- Collapse can be **silent on training reward** (rewards stay positive on easy prompts even as harder prompts fail). Monitor structured-output validity, not just reward.
- Higher learning rates accelerate collapse; lower LR delays but doesn't prevent it.
- The capability persists in the *weights* — early-stop the run before collapse and you still have a usable model, but you've left RL gains on the table.
- Generalizes across format and content OOD differently; pick the supervisory signal based on which OOD axis matters for your deployment.
- Code: [Tool-RL-Box](https://github.com/hypasd-art/Tool-RL-Box) reproduces the failure and the mitigations.

## Sources

- Paper: *Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It* — Hao, Jin, Liao, Liu, Zhao, 2026 — [arXiv:2606.26027](https://arxiv.org/abs/2606.26027). CAS Institute of Automation / UCAS.
- Code: https://github.com/hypasd-art/Tool-RL-Box
