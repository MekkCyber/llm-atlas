# Dual-Mode Chain-of-Thought
*Depth — training a single model to run both a fast (short-answer) and a slow (long-CoT) reasoning path.*

**TL;DR:** Production LLM traffic mixes routine queries (chatty, low-latency) with occasional hard multi-step problems that benefit from long chain-of-thought. Running two separate models is expensive; running one long-CoT model on everything wastes tokens on easy prompts. **Dual-mode CoT** trains a single set of weights to expose two reasoning regimes — fast thinking and slow thinking — dispatched by a prompt-level signal (system tag, control token, or learned complexity classifier). Training uses a **two-stage RL** that shapes each mode with its own reward. Hunyuan-A13B is the first frontier open-weights entry to formalize this as its primary post-training story.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [grpo](../grpo.md), [long2short](long2short.md)
**Related:** [prm](prm.md), [orm](orm.md), [length-penalty](length-penalty.md)

---

## What it is

A post-training scheme that yields **one model with two behaviors**, selectable at inference:

- **Fast thinking** — the model emits a short, direct answer (possibly with brief reasoning). Latency-sensitive.
- **Slow thinking** — the model emits a long, structured chain-of-thought before the final answer. Accuracy-sensitive on multi-step problems.

Both modes share weights. Dispatch is a *conditioning* mechanism — a system prompt tag, a control token, or an internal complexity classifier — not a router across models.

## How it works

**Two-stage RL** applied after a shared SFT stage:

1. **Reasoning-RL stage.** [GRPO](../grpo.md) with outcome-reward models (or a rule-based verifier where available) and, for code, sandbox execution feedback. Rollouts are long-CoT and rewarded on final-answer correctness. This is the training signal for the slow-thinking path. Reward is verifiable ([RLVR](../rlvr.md)) where possible; otherwise a learned outcome RM (see [orm](orm.md)).

2. **General-RL stage.** Same policy is now trained on a broader distribution — creative writing, chat, agent tasks, knowledge QA — using generative reward models that compare rollouts against reference answers, plus domain-specific reward services. Rollouts here are shorter; the reward penalizes over-length responses to prevent the slow-mode habit from leaking into casual replies.

The two stages together shape a policy that has learned when long-CoT pays and when it doesn't. Mode selection at inference is by conditioning (a system prompt like "think step by step" or a control token) that the model has learned to associate with each regime's training distribution.

## Why it matters

Adaptive reasoning depth was previously solved with either:
- **Two separate models** — expensive to serve, requires an external router (e.g. DeepSeek-V3 as the general-purpose model + R1 as the reasoner).
- **One reasoning model with a hard length cap** — wastes tokens on easy prompts because the model always kicks off long-CoT.
- **[Long2short](long2short.md) compression** — post-hoc distillation of long traces into short ones; works but loses the option to escalate on hard prompts.

Dual-mode CoT keeps a single-model deployment while preserving both regimes as first-class behaviors — a training-time answer to what has been a serving-time and orchestration-time problem.

## Gotchas & tricks

- **Reward-signal contamination.** If both stages use the same reward source, the model collapses toward whichever mode the reward prefers. The paper's two-stage split with different reward types (outcome sandbox for reasoning, generative RM for general) is what keeps the two behaviors distinct.
- **Dispatch reliability.** Prompt-tag dispatch is only as good as the model's compliance rate. Practical deployments often add an internal complexity classifier + slow-mode fallback on user demand.
- **Length-penalty in the general stage** matters — without it, the model over-generalizes long-CoT to routine queries and latency regresses.
- Similar in spirit to [long2short](long2short.md), but training-time and bidirectional (long2short only compresses; dual-mode preserves both).

## Sources

- Paper: *Hunyuan-A13B Technical Report* — Tencent Hunyuan Team, 2026 — [arXiv:2609.27284](https://arxiv.org/abs/2609.27284)
