# Sparse Context Routing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An inference-time attention pattern for models that carry growing history (chat, video generation, interactive editing): cache **reusable context states** and route a **fixed budget** of context tokens through attention at every denoising or decoding step. Per-step cost stays O(1) in history length. Introduced in ContextMaster for interactive multi-shot video creation; the pattern generalizes to any long-history generative UI.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md).
**Related:** [../multimodal/README.md](../multimodal/README.md) · [README.md](./README.md) · [../post-training/reasoning/long2short.md](../post-training/reasoning/long2short.md)

---

## What it is

Interactive generative sessions expand history over time: another chat turn, another shot in a video, another edit on the canvas. Attending densely to full history at every step is O(history × step-count). Sliding windows lose context; landmark tokens don't always cover what's needed. Sparse context routing addresses this by learning a **routing head** that picks a bounded subset of cached context tokens per attention step.

## How it works

**Cache.** Each completed history segment (a past shot, past turn, past canvas state) is encoded into a **clean context state** — a compact set of key/value pairs kept for later.

**Router.** At each denoising/decoding step, a small router head scores every cached context token against the current query and selects the top-$B$ tokens for attention. $B$ is a **fixed budget** — independent of history length.

**ConstraintSink.** Reserved routing slots that always attend to the current task's constraints (target prompt, style specs). Prevents the router from evicting the most task-critical context when it competes with historical content.

**Training.** Because sparse selection is non-differentiable, the router is trained via **privileged context distillation**: a dense-attention teacher provides target behavior; the sparse student is trained (consistency distillation, then distribution-matching refinement) to match. Two-stage training keeps quality high while the student learns to route.

## Why it matters

- **Per-step attention cost is O(B), not O(history).** Scales to hundreds of turns / shots without slowdown.
- **Preserves task constraints under history growth.** ConstraintSink solves the "the model forgot what I asked for" pathology of naive routing.
- **General primitive for interactive generative UIs.** The recipe transfers to any setting where a model must serve real-time interaction against a growing session state.
- ContextMaster reaches 16 FPS on a single GPU for interactive multi-shot video creation with cross-shot consistency.

## Gotchas & tricks

- **Router quality dominates.** A bad router evicts the wrong thing and quality drops sharply. Training the router with a competent dense teacher matters — no shortcut to random selection.
- **Budget vs quality tradeoff.** Smaller $B$ = faster but more evictions. Task-dependent sweet spot; run an ablation curve.
- **Constraint drift.** Even with ConstraintSink, if the task constraint is long (e.g. a full spec), routing over just the constraint tokens can miss subclauses. Chunk long constraints into multiple sink slots.
- **Cache invalidation is subtle.** If a past shot is edited, its cached state must be refreshed — silent staleness is a common bug.
- **Not for one-shot generation.** Overhead of maintaining the cache/router only pays off in genuinely interactive sessions.

## Sources

- Paper: *ContextMaster: Interactive Multi-Shot Video Creation via Fixed-Budget Sparse Context Routing* — Wei et al., Tsinghua / Kling / Kuaishou, 2026 — [arXiv:2608.04956](https://arxiv.org/abs/2608.04956). Introduces the routing recipe and privileged context distillation.
