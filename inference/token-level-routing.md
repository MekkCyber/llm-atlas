# Token-Level LLM Routing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Treat per-token model selection ("send this next token to the big model, that one to the small") as a *scheduling* problem rather than a prompt-time choice. TokenRouter (2026) factors a serving stack into per-model subservers that pull their work from per-request routing logic; a delayed-batching scheduler decides when each subserver should flush, tuned by a throughput model. Measured decoding-throughput gains of 2.01×–64.15× over single-model-oriented stacks.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [specfold.md](specfold.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Modern cost-reduction tricks — speculative decoding, small-model drafts, dynamic LoRA swap, hybrid big/small routing — all generate token-level decisions about which model to run. Classical LLM-serving systems (vLLM, SGLang, TRT-LLM) assume one base model and get hurt by two phenomena when a request routes mid-generation:

1. **Step desynchronization** — different models produce tokens at different rates; the shared batch stalls on the slowest.
2. **Batch admission delay** — when the chosen model's queue is short, waiting for a reasonable batch adds latency; when it's long, picking a different model would have been cheaper.

Token-level routing as a first-class scheduler contract fixes both.

## How it works

- **Programmable routing contract.** Each request carries a small routing-logic snippet (choose model as a function of produced tokens, costs, or external signals). The runtime treats it as a black box called between tokens.
- **Per-model subservers.** Each candidate model is served by its own subserver (its own batch, scheduler, KV cache). Subservers do not share batches across models.
- **Delayed-batching scheduler.** For each subserver, a throughput model predicts whether waiting for one more arrival yields net latency improvement. The scheduler flushes early when waiting is a net loss.
- **Cross-subserver flow.** A request's token stream is handed from one subserver to another as the routing logic decides; the stream, not the batch, is the unit of state.

## Why it matters

- Removes the implicit assumption "one request = one model" that bakes performance ceilings into every multi-model serving story.
- Lets researchers prototype routing policies (speculative tiers, expert-selection, LoRA swap) without re-plumbing a serving stack per idea.
- Reported throughput wins (2.01× worst, 64.15× best) suggest the scheduler layer, not the models, was the live bottleneck.

## Gotchas & tricks

- The throughput model is empirically tuned; wrong priors cause the scheduler to flush too eagerly or hang on empty queues.
- Routing logic that queries the state of distant subservers introduces coordination overhead — keep per-token decisions local.
- Fairness across requests is explicit, not automatic: a request with aggressive routing can starve a steady-stream one if subserver queues are shared.

## Sources

- Paper: *TokenRouter: Efficient Serving System for Token-Level LLM Routing* — Fu, Liu, Wang et al., Tsinghua-NICS-EFC, 2026 — [arXiv:2610.12242](https://arxiv.org/abs/2610.12242).
