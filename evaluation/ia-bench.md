# IA-Bench (Image Agent Bench)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for **agentic** image generation. Tests four core image-agent capabilities — **Plan, Reason, Search, Memory** — on real-world prompts that are underspecified, implicit, or require up-to-date world knowledge. Introduced alongside Qwen-Image-Agent (2026) to drive evaluation past "did the T2I model render a pretty picture" toward "did the agent construct enough context to render the *right* picture."

**Prereqs:** none
**Related:** [gauntletbench](gauntletbench.md), [coffeebench](coffeebench.md), [../multimodal/danceopd.md](../multimodal/danceopd.md)

---

## What it is

Standard image-generation benchmarks (GenEval, T2I-CompBench) hold the *prompt* constant and score the *rendering*. They assume the user prompt fully specifies the intended image. Real prompts don't — they're terse, omit constraints, reference unstated taste, or assume real-world facts the model has no fresh access to.

IA-Bench reframes the benchmark target. The system under test isn't a frozen T2I model; it's an **agent** that may plan, reason about missing context, search for facts, remember earlier interactions, and call the T2I model only after the context is sufficient. Capability axes:

| Axis | What it tests |
| --- | --- |
| **Plan** | Decompose an underspecified prompt into sub-goals + the order to satisfy them |
| **Reason** | Infer implicit constraints (style consistency, scale, lighting) from partial context |
| **Search** | Retrieve up-to-date facts (people, events, products) the base model doesn't know |
| **Memory** | Use prior interactions / generated outputs to drive consistency in later renders |

## How it works

1. **Prompts.** Carefully selected real-world image-generation requests that exhibit at least one of the four challenges (typically multiple).
2. **Agent under test.** An agent system that may call a T2I model (and optional tools — search, memory store, feedback loop) one or more times.
3. **Scoring.** Image fidelity + capability-axis-specific checks (did the agent's reasoning trace identify the missing context? did the searched facts get incorporated? did consistency across turns hold?).
4. **Baselines.** Frozen T2I models and naive "rewrite-the-prompt" agents form the bottom of the leaderboard; multi-tool agents like Qwen-Image-Agent occupy the top.

## Why it matters

- **Closes a real evaluation gap.** Frontier T2I models keep "improving" on benchmarks while *production* image-agent UX still feels brittle — the gap is *context construction*, not rendering. IA-Bench measures the right thing.
- **Pairs with image-agent training.** A target for any system trying to train T2I tool-use end-to-end. As the field moves from frozen-T2I + naive prompt rewrite toward true agentic image generation, IA-Bench is the obvious benchmark to optimize.
- **Generalizes the pattern.** The Plan / Reason / Search / Memory axes aren't image-specific — same framework extends to agentic video generation, agentic 3D modeling, and other open-ended multimodal output.
- **Complements [GauntletBench](gauntletbench.md)** (computer-use agents) and [CoffeeBench](coffeebench.md) (economic agents) along the agent-evaluation axis.

## Gotchas & tricks

- **Search retrieval is hard to grade fairly.** Some prompts require facts no agent could know at the cutoff date; the benchmark must distinguish "search failure" from "fact didn't exist."
- **Memory tests can leak.** Many memory subtasks share assets across turns; an agent that memorizes the benchmark wins without true memory. The paper has held-out variants.
- **Capability-axis decomposition is approximate.** Real prompts trigger multiple axes; per-axis scoring is a useful directional signal but not a clean factorization.
- **The "agent" abstraction matters.** Comparing two agents is comparing their entire stacks (tools, schedulers, memory) — not just the underlying T2I weights. Report per-stack details.
- **Score is sensitive to T2I backbone.** Even a perfect context-construction agent caps out at whatever the underlying renderer can produce. Don't conflate agent quality with renderer quality.

## Sources

- Paper: *Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation* — Zhang, Li, Zhang, Gao, Yan, Jiang, Tang, Yin, Wu, Chen, Xu, Shu, Zhang, Xu, Chen, Wang, Liu, Zhou, Zhang, Zhao, Wu, 2026 — [arXiv:2606.26907](https://arxiv.org/abs/2606.26907) — Qwen / Alibaba.
- Background: *GenEval / T2I-CompBench* — prior text-to-image benchmarks that hold the prompt constant.
