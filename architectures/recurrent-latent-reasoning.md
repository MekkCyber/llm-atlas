# Recurrent Latent Reasoning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Do the "reasoning" step *inside* the model's hidden state instead of verbalizing it as tokens. Inputs continuously update a recurrent memory as they arrive; queries are answered by iterating a solver in a high-dimensional latent space, with no intermediate CoT emitted. Popularized by BDH-CQ (2026): a 150M-parameter reasoner hits 29.5% pass@2 on ARC-AGI-1 for $0.0007/task — a new cost/accuracy Pareto point for pattern-transformation reasoning benchmarks.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [multi-head-attention.md](multi-head-attention.md)

---

## What it is

Long chain-of-thought reasoning (o1 / R1 / long-CoT-RL) treats *tokens* as the substrate for intermediate reasoning steps. Every reasoning step costs tokens and inference latency; token count is the scaling knob. Recurrent latent reasoning is the alternative bet: keep the intermediate steps in the hidden state, don't verbalize them, and pay per **latent iteration** instead of per **emitted token**.

Two structural pieces:

1. A **recurrent memory** that ingests context (demonstrations, prior turns, the current question) and updates its state at each input. Analogous to an SSM / RWKV / RNN-style summary.
2. An **iterative latent solver** that answers a query by repeatedly updating a latent vector — a fixed-depth loop of transformer-like layers applied to the state until a stopping criterion (fixed iterations, convergence, or a learned halt signal).

The emitted output is just the final answer; the "reasoning" that produced it lives entirely in the latent space.

## How it works

Idealized decode loop for a query `q` given memory state `s`:

```
z_0 = q_encoder(q, s)                 # inject question into latent
for i in 1..K:                         # K = iteration budget
    z_i = solver(z_i-1, s)            # iterated latent update
answer = decoder(z_K)                  # verbalize only the final answer
```

The `solver` step is where "reasoning" happens. Because it operates in latent space, it doesn't pay a per-step token cost — the model's compute is proportional to `K × (per-iteration-flops)`, not to any generated CoT length. `K` is the reasoning-depth knob, analogous to token count for long-CoT models.

Training is typically self-supervised with an outcome loss on the final answer, sometimes with a differentiable halt signal to learn `K` dynamically.

## Why it matters

- **Decouples reasoning cost from output length.** Long-CoT costs scale linearly with CoT length in tokens; latent reasoning costs scale with `K` in latent iterations. `K` can be much smaller and more tunable than reasoning-token counts.
- **Enables tiny reasoners.** BDH-CQ demonstrates the extreme: 150M parameters, sub-cent per task, competitive on ARC-AGI-1. Frontier long-CoT models cost orders of magnitude more per task at similar accuracy.
- **Test-time compute knob without token inflation.** Increase `K` to spend more compute on hard queries without changing the model or the prompt structure.
- **Aligns with SSM / linear-recurrence research.** The recurrent-memory piece is a natural home for SSM / Mamba-style state updates; the solver is a natural home for weight-tied transformer iteration or diffusion-style refinement.

## Gotchas & tricks

- **Interpretability is worse than CoT.** With verbal CoT you can inspect the reasoning; latent-reasoning traces are opaque high-dimensional vectors. Diagnostics need probing tools.
- **Task generalization is unproven at scale.** ARC-AGI is pattern-transformation, well-suited to a latent-iteration substrate. Whether the recipe transfers to open-ended math, code, or knowledge-heavy reasoning is the open question.
- **`K` interacts with training curriculum.** Train with `K` too small and the model never learns to use iterations; too large and each iteration underspecializes. Common recipe: schedule `K` upward during training.
- **No natural "reasoning trace" for CoT monitoring.** Downstream safety measures that rely on inspecting CoT (e.g. cot-monitoring) don't apply — the analog would be latent-state probing.
- **Cost/accuracy Pareto shifts by domain.** The BDH-CQ ARC-AGI numbers set a new frontier on that benchmark; on token-rich reasoning tasks, verbal CoT may still win.

## Sources

- Paper: *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning* — 2026 — the reference open realization.
- Background: SSM / Mamba literature for the recurrent-memory side.
- Contrast: long-CoT RL — [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) — the counterpoint that keeps reasoning in token space.
