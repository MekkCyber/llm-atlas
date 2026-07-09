# Thinking mode
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-time integrated reasoning trace: the model learns to emit a chain of internal reasoning before the user-visible answer, and conditions its answer on that trace. Not a post-hoc CoT prompt wrapper — thinking mode is baked into the weights so the model can be *toggled* into extended reasoning at inference. o1 and R1 pioneered the pattern; Kimi-k1.5 and now Gemma 4 land it as a standard feature across a model family.

**Prereqs:** [long-cot-rl.md](./long-cot-rl.md), [../grpo.md](../grpo.md)
**Related:** [length-penalty.md](./length-penalty.md), [long2short.md](./long2short.md), [../../case-studies/deepseek-r1.md](../../case-studies/deepseek-r1.md), [../../case-studies/kimi-k1-5.md](../../case-studies/kimi-k1-5.md), [../../case-studies/gemma-4.md](../../case-studies/gemma-4.md)

---

## What it is

Chain-of-thought prompting ("think step by step") is a *prompting* technique — the model conditions its output on a natural-language instruction to reason. Thinking mode is a *training* technique — the model has been post-trained (SFT + RL) to emit reasoning traces natively, conditioned on nothing more than the presence of a mode flag or a task type.

At inference, thinking mode has a distinct output structure: the model emits a **reasoning trace** (visible or hidden depending on product design) → a **final answer** conditioned on that trace. In modern systems the trace can be arbitrarily long (10K+ tokens for hard reasoning tasks) and is what actually drives the answer quality.

## How it works

**Structure at inference.** The model produces two segments:

```
<think>
… multi-thousand-token internal reasoning trace …
</think>
<answer>
… final user-visible answer …
</answer>
```

Delimiters and formatting vary per model (Gemma 4, R1, Kimi-k1.5 use different tokens). Both segments come from the same forward pass — the model just self-conditions on its own reasoning.

**Training pipeline.** Two-stage:

1. **SFT on reasoning traces.** Collect high-quality (question → long reasoning trace → correct answer) triples. Sources: expert-written, distilled from stronger models, or generated then filtered by outcome. Train the model to emit both segments.

2. **RL with verifiable rewards.** On reasoning-heavy tasks (math, code, logic), run RL where the reward comes from checking the *final answer* against a rule (correct number, passing tests). Trace tokens are unrewarded per-token but shape the answer distribution via credit assignment. GRPO / mirror-descent variants both work.

Optionally: a length penalty ([length-penalty.md](./length-penalty.md)) to prevent unbounded trace inflation, or long-to-short distillation ([long2short.md](./long2short.md)) to compress traces post-hoc.

**Toggling.** Systems expose thinking mode as an on/off switch (Gemma 4's stated design) or an implicit trigger based on task complexity. When off, the model skips the trace segment and answers directly — useful for simple queries where reasoning is wasted compute.

## Why it matters

- **State-of-the-art on reasoning benchmarks** across math, code, competition problems. Every top open reasoning model in 2025–2026 has some version of thinking mode.
- **First-class output structure.** Users can inspect the reasoning; interpretability research can attach to it; safety-critical applications can audit it.
- **Consolidation of the design pattern.** o1 introduced it in closed form; R1 open-sourced it; Kimi-k1.5 refined the RL; Gemma 4 lands it in an open-weight *family* across sizes. The pattern has crystallized.
- **Compute-elastic quality.** Same model, same weights — thinking on for hard problems, thinking off for cheap ones. Users choose their quality/latency point per query, not per model.

## Gotchas & tricks

- **Reasoning traces can be reward-hacked** — the model learns to emit trace patterns the RL judge likes without them actually improving the answer. Verifiable rewards (rule-based) resist this; model-judged rewards are vulnerable.
- **Trace-length inflation is a real problem.** Without length regularization, traces grow unbounded during RL. Adding a per-token cost or a KL-to-shorter-baseline term helps ([length-penalty.md](./length-penalty.md)).
- **Users may not see the trace.** Product designs vary — OpenAI hides most of it, DeepSeek shows it, Gemma 4 is TBD. Design choice, not a training constraint.
- **Toggling requires a training signal for "no-think" mode.** If all training data has traces, the model can't be reliably switched to no-think mode without extra data.
- **Not the same as CoT prompting.** CoT prompting works on any model; thinking mode requires training. The gap between them is roughly the same as prompting vs fine-tuning.
- **Composes with distillation.** Long2short ([long2short.md](./long2short.md)) compresses traces from a thinking-mode model to yield a faster inference-time model that keeps the answer quality.

## Sources

- *OpenAI o1 System Card* — OpenAI, 2024 — first public thinking-mode model.
- *DeepSeek-R1* — DeepSeek, 2025 — open thinking-mode training pipeline. See [../../case-studies/deepseek-r1.md](../../case-studies/deepseek-r1.md).
- *Kimi k1.5* — Moonshot, 2025 — mirror-descent RL for reasoning. See [../../case-studies/kimi-k1-5.md](../../case-studies/kimi-k1-5.md).
- *Gemma 4 Technical Report* — Google DeepMind, 2026 — [arXiv:2607.02770](https://arxiv.org/abs/2607.02770) — integrated thinking mode across model family.
