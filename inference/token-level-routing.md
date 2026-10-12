# Token-Level LLM Routing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of picking one model per request, pick a model per **token**: route "hard" tokens to a strong (slow) model and "easy" tokens to a cheap (fast) one. The traditional obstacle isn't accuracy, it's serving — token-level routing shatters normal batches. TokenRouter solves it with request-centric programming + model-centric execution and a delayed-batching scheduler, reporting 2.01×–64.15× decoding-throughput wins over existing systems.

**Prereqs:** [README.md](./README.md), [../architectures/_moe.md](../architectures/_moe.md)
**Related:** [../architectures/aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md)

---

## What it is

A serving-system design for mixed-model inference where the choice of which model emits each token is made *during decoding*, not at request admission. The programming model exposes per-token routing as a developer-written callback; the runtime deals with the batching nightmare that creates.

## How it works

Two separations do the work:

1. **Request-centric programming / model-centric execution.** The user writes "which model should produce this token" at the request level; the system runs one *subserver* per candidate model and only sees "token X of request Y goes to model M."
2. **Delayed batching.** Rather than dispatching each routed token immediately (which would starve every model into batch-of-one), the scheduler holds routed tokens briefly to form homogeneous micro-batches per model, trading a small latency floor for a large throughput floor.

Attention and KV-cache state are kept per-request-per-model; cross-model handoff happens at the token boundary, so a strong-model's KV cache survives across its own invocations even when a cheap model fills in between.

## Why it matters

Token-level routing has been a research idea for years and a serving dead-end for just as long: standard LLM servers assume a request lives on one model, so per-token model swaps collapse batches. TokenRouter removes that assumption at the scheduler layer, making cascaded inference, expert-choice routing, and speculative-plus-routed hybrid stacks actually deployable. The reported 2.01×–64.15× speedups are at **unchanged task quality**, i.e. the win is pure systems.

## Gotchas & tricks

- Delayed-batching latency is paid on *every* model switch. Policies that switch frequently are penalised; policies that stick with one model for long runs are near-free.
- KV-cache memory grows roughly linearly with the number of models each request may touch — plan for it when sizing the serving fleet.
- Routing policy quality is orthogonal to the serving system. TokenRouter speeds up any policy; it does not learn one. Pair with a trained router (e.g. logit-margin or confidence-threshold).
- The 64× number comes from workloads that previous systems serialised; typical mixed workloads land in the 2–5× range.

## Sources

- Paper: *TokenRouter: Efficient Serving System for Token-Level LLM Routing* — Fu et al., Tsinghua NICS-EFC, 2026 — [arXiv:2610.12242](https://arxiv.org/abs/2610.12242)
