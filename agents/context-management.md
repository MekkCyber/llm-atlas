# Model-Managed Context
*Depth — expose a context-manipulation API to the LLM so the model (not the harness) decides when and what to compact, with KV-cache reuse across edits.*

**TL;DR:** The standard LLM-agent harness compacts conversation history when the context window fills. Model-managed context flips this: the model is given a **first-class API** for context operations (compact, pin, drop regions) and decides when to invoke them. The NVIDIA 2026 study finds that model-driven management often beats harness-driven management on accuracy — sometimes with *fewer* input tokens — with the largest gains at tight budgets. A side result: reusing the KV cache of unchanged context after an edit cuts prefill compute by up to **15% per request** at no measurable accuracy loss.

**Prereqs:** [README](README.md), [../inference/README.md](../inference/README.md)
**Related:** [memory-agents](memory-agents.md) · [../inference/kv-cache-reuse.md](../inference/kv-cache-reuse.md) · [rubric-based-credit](rubric-based-credit.md)

---

## What it is

In harness-managed context, a loop outside the model watches the token count and applies a compaction rule (summarize, truncate oldest, drop tool outputs) when the window fills. The model has no say in *when* or *what* is compacted.

In model-managed context, the harness exposes functions the model can call:

| API call | Effect |
| --- | --- |
| `compact(region)` | Replace a span with a model-written summary |
| `pin(region)` | Mark a span as never-compactable |
| `drop(region)` | Remove a span entirely |
| `restore(region)` | Expand a previously-summarized span |

The model interleaves these calls with normal tool calls. Context management becomes part of the agent's reasoning rather than external housekeeping.

## How it works

At each step of the agent loop, the model sees its current context plus the context API as part of its tool schema. When context usage grows (or the model anticipates it will — e.g. before a long tool call), it emits a `compact` or `drop` call in the same syntactic form as any other tool call. The harness applies the edit and continues.

The KV-cache reuse trick:

- Before the edit, the harness has a KV cache for the full context $C$.
- After the edit, the context is $C' = C_{\text{prefix}} \oplus \text{edit} \oplus C_{\text{suffix}}$.
- The prefix portion of the KV cache is still valid; only the edited region and everything after it need re-prefill.
- The paper reports up to **15% prefill savings per request** with no accuracy loss when this reuse is implemented cleanly.

## Why it matters

- **Settles a live architectural question.** "Should the harness or the model own context management?" has been argued for both sides. The paper gives a concrete answer: **give the model the API**, especially at tight budgets (8K–64K tokens) and for tasks with heavy tool output.
- **Accuracy *and* efficiency.** Model-driven management sometimes uses *fewer* input tokens while scoring higher. The model's context sensitivity to the current task is a better compaction signal than the harness's generic rules.
- **KV-cache reuse becomes routine.** Editing context mid-task without re-prefilling the whole thing was previously a research curiosity; the paper's results make it a production optimization.

## Gotchas & tricks

- **Behavior varies across models.** Different models use the API wildly differently — some compact aggressively, some never call `compact` and just let the harness fall back. The paper documents this variance but doesn't fully explain it.
- **Needs training or careful prompting.** Out-of-the-box models often misuse the API (compact too aggressively, pin irrelevant regions). Either RL-train the API usage against downstream reward (see [rubric-based-credit](rubric-based-credit.md)-style rewards) or constrain the API in the system prompt.
- **Edits can break downstream references.** If the model later tries to reference a dropped region (e.g. a tool call's output), it can't. Harness-level guards that reject `drop` on regions still referenced by the current reasoning trace prevent this.
- **KV reuse requires position-stable edits.** Reuse works cleanly when edits are either append-only or replace a contiguous span whose length doesn't change. Arbitrary insertions in the middle invalidate position embeddings downstream — the paper's 15% gain assumes edits respect this.

## Sources

- Paper: *Letting Models Manage Their Own Context: Where We Stand* — Devoto, Vem, Safavi, Kariyappa, Tseng, Furgale, Silveira Cabral (NVIDIA), 2026, arXiv 2611.34147.
- Related: [memory-agents](memory-agents.md) — companion pattern that *externalizes* context into a separate memory agent instead of editing in place.
- Related: [../inference/kv-cache-reuse.md](../inference/kv-cache-reuse.md) — the underlying KV-cache-edit mechanics.
