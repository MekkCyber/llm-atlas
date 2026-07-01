# Tool-Use Credit Assignment
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In agentic RL with code/tool calls, outcome rewards are too coarse to say *which* tool call helped, and process rewards need an external judge. **TACO** proposes a self-supervised alternative: a **Differential Answer-Probe Reward (DAPR)** that measures each call's marginal effect on final correctness via probe tokens, plus **Outcome-Gated Advantage Routing (OGAR)** that routes final-answer credit only to the responsible segments. Drops into GRPO as a variant, no judge model required.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md), [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [README.md](./README.md)

---

## What it is

A tool-augmented agent produces a trace of the form
`reason → tool_call_1 → observe → reason → tool_call_2 → … → answer`.
The final reward only says whether the answer was right. Two failure modes are impossible to distinguish from that signal alone: (a) the answer was right *because of* a specific tool call, and (b) the answer was right *despite* a redundant or misleading call. Standard PRMs solve this with an external judge that scores each step — expensive, and the judge itself has a distribution.

TACO removes the judge. Each tool call gets its own advantage from a self-consistency probe against the agent itself.

## How it works

**DAPR — Differential Answer-Probe Reward.**
Insert probe tokens into the agent's reasoning trace that force it to commit to an intermediate predicted answer *with* and *without* a given tool call's observation. Read the delta:

```
call_credit_i = R(answer | with tool_i) − R(answer | without tool_i)
```

Positive → useful; negative → misleading; zero → wasteful. Reusing the *existing* verifier for both probes means no auxiliary judge. Being a **difference** of probe scores rather than an absolute one makes it naturally resistant to probe-hacking (a shift that boosts both terms cancels).

**OGAR — Outcome-Gated Advantage Routing.** A parameter-free rule that gates the final-answer credit by the outcome:
- If the answer is correct, distribute the final-answer advantage only to segments preceding *useful* tool calls (positive DAPR).
- If the answer is wrong, negative advantage lands only on segments containing *misleading* calls (negative DAPR).
- Wasteful segments (DAPR ≈ 0) receive zero credit either way — implicit anti-redundancy pressure without any explicit cost term.

**Training pipeline.** SFT → RL. The RL stage uses GRPO with two coupled advantage channels (DAPR + OGAR-routed outcome). Group-relative normalisation is applied independently within each channel.

## Why it matters

- **Self-supervised process reward** for tool use — no judge model, no annotation. First serious alternative to PRM-family process rewards in an agentic setting.
- **Cuts redundant tool calls** without a hand-tuned cost regulariser. The model learns to skip tools that DAPR reports as ≈ 0.
- **Composes with any GRPO-family recipe.** The advantage channels sit in the same slot as the standard reward; existing rollout infrastructure applies unchanged.

## Gotchas & tricks

- **Probe placement matters.** Probes that trigger too early get uninformative deltas; too late and the tool call is already consumed. Paper places them at each reasoning-to-tool boundary.
- **DAPR requires two forward passes per probe.** Cost grows linearly in the number of tool calls; batching probes helps but doesn't remove the tax.
- **Probe-hacking is bounded, not solved.** A model can still shift both probe outputs together; the paper reports this is rare with GRPO's KL to the reference, but expect it as `β` shrinks.
- **Extends beyond multimodal.** Paper reports results on perception, reasoning, and general multimodal benchmarks; DAPR/OGAR are text-only mechanisms that transfer to any code-tool agent.

## Sources

- Paper: *TACO: Tool-Augmented Credit Optimization for Agentic Tool Use* — Feng et al., 2026 — [arXiv:2606.30251](https://arxiv.org/abs/2606.30251).
