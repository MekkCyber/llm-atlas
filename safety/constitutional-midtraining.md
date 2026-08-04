# Constitutional Midtraining
*Depth — durable alignment via constitutional content injected during midtraining, not post-training.*

**TL;DR:** Post-training alignment is shallow — SFT and benign fine-tuning erode it quickly. Constitutional midtraining injects a **394M-token constitutional corpus** into the midtraining stage (before instruction tuning), at 120B scale, and produces alignment gains that survive downstream SFT and even benign fine-tuning. On blackmail propensity specifically, the advantage persists at **−17.5 pp** after benign fine-tuning. Curriculum ordering and deliberative-reasoning traces both help, but the primary driver is simply *content presence*.

**Prereqs:** [mid-training.md](../pre-training/mid-training.md)
**Related:** [alignment-faking.md](./alignment-faking.md), [sleeper-agents.md](./sleeper-agents.md)

---

## What it is

An intervention placed *between* pretraining and post-training that pre-installs values-based knowledge and behavior before the fragile alignment layer is added on top. The hypothesis: alignment fails to durable fine-tuning because it's a shallow, late-stage modification; if the constitutional content is present in the model's midtraining distribution, downstream fine-tuning can't easily overwrite it.

## How it works

1. **Build a constitutional corpus** — the paper uses ~394M tokens derived from Anthropic's Constitution, restructured into training-appropriate documents.
2. **Run midtraining** at 120B scale with a 2×2 factorial design:
   - **Curriculum ordering:** constitutional content interleaved vs. clustered.
   - **Deliberative reasoning:** raw constitutional principles vs. principles + worked-through reasoning traces.
3. **Compare** against a replay-only control (same token budget, no constitutional content).
4. **Evaluate at three stages:** post-midtraining, post-SFT, and post-benign fine-tuning.
5. **Benchmarks:** self-generated + established alignment probes (blackmail, alignment-under-pressure, value-conflict resolution, emergent misalignment) plus capability probes (MMLU, ARC-Easy, PIQA, GSM8K).

## Why it matters

- Alignment stops being a fragile last-step veneer if a modest midtraining intervention can make it durable through downstream fine-tuning.
- The 2×2 factorial gives a rare, clean answer to a persistent question: **content presence dominates**; structure (curriculum ordering, deliberative reasoning) is second-order.
- No capability tax on the tested benchmarks at any stage — free lunch, at least on these measures.

## Gotchas & tricks

- Durability is *not* uniform: settings requiring active resistance to in-context pressure or explicit value conflicts see the advantage attenuate after SFT. Constitutional midtraining is stronger on **latent propensities** (blackmail) than on **active reasoning** under pressure.
- 394M tokens is a rounding error at 120B midtraining scale — but it's the *presence* in the distribution that matters, not the volume. Larger doses may not scale linearly.
- Not a substitute for post-training alignment; a complement to SFT-centered pipelines.

## Sources

- Paper: *Constitutional Midtraining: Content Presence Drives Alignment Gains* — Cho et al., 2026 — [arXiv:2607.26654](https://arxiv.org/abs/2607.26654)
