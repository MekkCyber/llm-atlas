# Tool-Aware Scheduling (TAO)
*Depth — a joint scheduler for LLM agentic serving that admits sessions on near-term tool-return readiness, KV-cache credit, and token footprint.*

**TL;DR:** TAO (Li et al., 2026) is a scheduler for LLM agent workloads that replaces heuristic GPU admission with a dynamic program over three per-session signals: (1) **near-term tool-return readiness** — probability or ETA of the session's next tool call completing, (2) a **time-decayed per-node KV cache credit** — how much warm cache the session still has on each GPU, (3) **token footprint** — current + expected context length. Across 12 (model, workload, concurrency) settings, it reports **up to 26.8% lower latency, 36.3% lower mean TTFT, and 29.5% higher decode throughput** vs. the best baseline on each metric.

**Prereqs:** [agent-serving.md](agent-serving.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

A joint admission + placement scheduler for a serving cluster that handles agent sessions. On each scheduling tick:

```
candidates = sessions in state WAITING_TOOL or READY
for each (session s, GPU g) pair:
    score(s, g) = w1 * readiness(s)
                + w2 * cache_credit(s, g)
                − w3 * footprint(s)
pick admit set via dynamic program subject to:
    per-GPU KV capacity
    per-GPU compute capacity
    at most one DECODING slot per batch per GPU
```

The output: which sessions decode next, on which GPUs. The dynamic program handles the KV-capacity constraint — a knapsack-flavored problem across sessions and nodes.

---

## How it works

### Readiness signal

For each `WAITING_TOOL` session, estimate the probability the next tool return will arrive within the current scheduling window. Sources:

- Historical latency of this tool class.
- Elapsed wait time (hazard rate rising with age for most tools).
- Explicit `expected_completion` hint from the tool when available (e.g., batch API ETAs).

The ETA is deliberately probabilistic. Point estimates fail: short-tail tool returns swamp any mean-based policy.

### KV cache credit

Each (session, node) pair accumulates a credit when the session's cache is on that node:

```
credit(s, g, t) = Σ (bytes_warm_on_g(s, τ) * decay(t − τ))
                τ ≤ t
```

Decay matters: a session that was decoded on node `g` an hour ago but has not been back has lower effective locality than one that returned 100 ms ago. The decay rate is a tuning parameter; TAO uses an exponential with half-life on the order of minutes.

### Token footprint

```
footprint(s) = current_context_tokens(s)
             + expected_remaining_tokens(s)
```

The second term is estimated from historical generation lengths per workload class. Footprint acts as a negative term — large sessions crowd out small ones unless their readiness or locality more than compensates.

### Dynamic program

The schedule-step problem is: pick a subset `A` of (session, GPU) pairs that maximizes `Σ_{(s,g) ∈ A} score(s, g)` subject to per-GPU KV capacity and compute slots. TAO solves this with a DP that exploits per-GPU separability after a sorting pass; wall-clock cost is a few milliseconds for cluster sizes considered in the paper.

---

## Why it matters

- **Three signals, one objective.** Prior work optimized each in isolation: continuous batching for throughput, prefix caching for locality, SLO queues for latency. TAO folds all three into one score, with weights that are workload-tunable.
- **Joint admission + placement.** Separating the two (admit by readiness, then place by locality) is strictly worse than solving them jointly, because admit decisions constrain placement decisions.
- **Headline numbers.** Up to **26.8% lower latency** across 12 model/workload/concurrency settings vs. per-metric best baselines. Not a point improvement — a Pareto shift.
- **Scheduler-only intervention.** No model changes, no kernel changes. Deployable as a replacement scheduling layer on top of vLLM / SGLang-style engines.

---

## Gotchas & tricks

- **Weight tuning is workload-specific.** `w1, w2, w3` for a browser-agent workload differ from a code-execution workload. The paper reports values per setting; redo for your own traffic.
- **Readiness miscalibration is the main failure mode.** Over-admitting sessions that aren't actually ready wastes slots. Under-admitting leaves GPU idle. The ETA estimator has to be measured, not borrowed.
- **Cache credit decay rate matters.** Too fast → thrashing. Too slow → stuck on cold cache after legitimate idleness. Tune per expected tool-latency distribution.
- **DP ignores cross-session cache sharing.** Shared system-prompt prefixes are modeled only coarsely. If prefix caching is dominant, pair with an orthogonal prefix-cache scheduler.
- **Starvation guard required.** Pure score-maximizing admission can indefinitely postpone high-footprint sessions. Add a hard age-based priority bump.

---

## Sources

- Paper: *TAO: Tool-Aware Joint Scheduling for LLM-Based Agentic Serving* — Li et al., 2026, [arXiv 2610.37230](https://arxiv.org/abs/2610.37230) — the method and the benchmark numbers.
- Related: [agent-serving.md](agent-serving.md) — the broader design-space TAO sits in.
- Related: [../systems/partial-rollouts.md](../systems/partial-rollouts.md) — the training-time analogue of session-aware scheduling.
