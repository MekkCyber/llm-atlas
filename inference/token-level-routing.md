# Token-Level LLM Routing
*Depth — switching models per-token inside a single decode to trade cost against quality.*

**TL;DR:** Token-level routing picks a different LLM for each decoded token, inside the same request, based on an inline router's confidence or difficulty estimate. The routing *quality–cost* tradeoff is strong in theory; the *serving* story is where most systems fall over, because single-model schedulers synchronise on one model and stall at every switch. TokenRouter (2026) is the first serving stack built for it, reporting **2.01× to 64.15×** throughput over existing systems.

**Prereqs:** [README.md](README.md)
**Related:** [_speculative-decoding.md](_speculative-decoding.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

An inference pattern where the decoder, for each next token, chooses which model in a pool actually produces it. Typical pools: one small fast model + one large accurate model, or a tiered mixture (nano → small → medium → large). A router — often a tiny MLP over the current hidden state and a difficulty signal — decides.

Not model cascading (which routes whole queries), not expert MoE (which routes tokens *inside* one model). Token-level routing routes across *independent* models.

## How it works

Three serving-level primitives (TokenRouter):

- **Request-centric programming, model-centric execution.** The user program describes a request's dataflow across models (a DAG of per-token model choices); the runtime maps each request to many per-model subservers, each of which batches requests locally.
- **Per-model subservers.** Every candidate model runs its own continuous-batching worker. A request's state (KV cache fragments, pending tokens) moves between subservers as the router decides.
- **Delayed-batching scheduler.** Rather than admit a token-generation step the instant it becomes schedulable, hold it briefly to let the batch fill. The hold time is picked from an explicit **throughput–latency model** so you don't trade too much tail latency for throughput.

The router itself can be learned (small classifier head) or heuristic (entropy of the small model's distribution, agreement with a draft, verifier score).

## Why it matters

- **Realises the paper gains.** Routing papers typically show offline quality–cost curves. Real serving systems rarely survived the switching overhead; TokenRouter shows the gains can live past the overhead with proper batching.
- **Composes with other tricks.** The same async per-model subserver pattern underlies speculative decoding (draft/verify), MoE expert parallelism, and prefill/decode disaggregation.

## Gotchas & tricks

- KV-cache migration across models is the single hardest part — cheap routers assume one logical cache per request, but token-level routing forces per-model partial caches. Design with explicit cache-slot ownership and copy-on-switch (or just accept duplicate caches).
- Delayed-batching trades tail latency for throughput; the throughput-model calibration must be redone whenever model-pool mix or hardware changes.
- Mis-routing cascades: once a wrong model generates a confident token, downstream decisions inherit it. Validate the router under distribution shift.
- Avoid routing-switch oscillation (ping-pong between two models for consecutive tokens) with a short-run hysteresis window.

## Sources

- Paper: *TokenRouter: Efficient Serving System for Token-Level LLM Routing* — Fu et al., Tsinghua NICS-EFC, 2026 — [arXiv:2610.12242](https://arxiv.org/abs/2610.12242).
