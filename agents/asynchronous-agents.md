# Asynchronous Agents
*Depth — a general framework letting one LLM run overlapping inference coroutines that share memory.*

**TL;DR:** Standard LLM agent loops are strictly sequential: read → think → reply or call tools → repeat. Real deployments — voice assistants, embodied agents, monitors, streaming video — receive new inputs *while the model is still thinking*. Yakushev et al. (Nebius / Yandex, 2026) propose an **asynchronous LLM framework**: users (or agents themselves) declare inference **coroutines** with overlapping KV-memory states, and off-the-shelf Qwen 3.x handles streaming video, videogames, and monitoring with no task-specific training.

**Prereqs:** [README.md](./README.md), [../fundamentals/dca.md](../fundamentals/dca.md)
**Related:** [context-language-models.md](./context-language-models.md), [agent-harness.md](./agent-harness.md), [../inference/README.md](../inference/README.md)

---

## What it is

Specialized async stacks exist today for each use case: voice-mode LLMs, VLMs for streaming video, VLAs for robots, async tool-calling APIs. They all share the same shape — one or more input streams whose arrival is independent of the model's current token position. The paper argues this shape is general and should be a first-class abstraction over the model's KV state, not a per-domain architecture.

## How it works

Three pieces:

1. **Inference coroutines.** Each coroutine is a separable generation trace with its own position in the prompt and its own stopping condition. A caller declares coroutines dynamically — e.g. one coroutine listens to the user, another reasons, a third drives the tool loop.
2. **Overlapping memory states.** Coroutines share (fully or partially) the same KV cache. A write by one coroutine (new tokens, pre-emptive summarization, context edits à la [context-language-models.md](./context-language-models.md)) becomes visible to the others under declared sharing rules.
3. **No task-specific training.** The paper's result is that Qwen 3.x already generalizes to this coroutine abstraction zero-shot. Streaming-video understanding, videogame play, and monitoring were run on stock weights.

## Why it matters

- **Collapses multiple bespoke stacks.** Voice, video, embodied, and monitoring agents all become harness configurations of one asynchronous runtime. Fewer separate training recipes to maintain.
- **Edge-latency friendly.** Overlapping memory means the model can keep "reading" the environment while it still "thinks" — the end-to-end latency bottleneck of sequential agent loops disappears.
- **Prerequisite for multi-sensor agents.** Any realistic embodied agent takes several simultaneous sensor streams; forcing them into a single sequential queue is the main reason current VLAs feel laggy.

## Gotchas & tricks

- **Coroutine coherence is on you.** If two coroutines write incompatible content to the shared KV, the model will blend them inside a single attention operation and produce incoherent output. Protocols (locks, append-only regions, per-coroutine partitions) matter.
- **Zero-shot ≠ no supervision.** Qwen 3.x generalizes, but it was trained with long multi-turn, tool-use, and streaming data. A model trained only on single-turn SFT will not pick up coroutine semantics for free.
- **Serving-side KV cache work.** Overlapping coroutines break the usual "one request, one KV cache" assumption. SGLang / vLLM extensions for shared/branched KV are needed to realize the win.
- **Coroutines interact with [context-language-models.md](./context-language-models.md).** CLM-style edits by one coroutine propagate across the shared file; useful for coordinating planners and executors, but demands clear ownership rules per span.
- **Benchmarks are per-domain.** Streaming video, videogames, and monitoring are evaluated separately; no unified "async agent" benchmark yet — this is a 2027 gap.

## Sources

- Paper: *LLMs are General Asynchronous Agents* — Yakushev, Mazur, Bartenev, Zhdanovskiy, Byzov, Kaurkin, Pastushenko — Nebius / Yandex, 2026 — [arXiv:2609.35427](https://arxiv.org/abs/2609.35427).
