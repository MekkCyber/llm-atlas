# Context Management for Agent Harnesses

*Taxonomy — strategies for keeping a long-horizon agent's context under the model's window.*

**TL;DR:** As an agent trajectory grows, the raw stream of observations, tool outputs, and reasoning quickly overruns the model's context window. A context manager decides what to drop, summarize, or keep verbatim on every step. The modern default is **staged rule-based elision followed by LLM summarization**: cheap deterministic cleanup first, model-side compression only when it's still overflowing. Recoverable elision (elided content that can be re-fetched on demand) sounds good but models rarely use it.

**Related taxonomies:** none yet. Sits under the [agent-harness](agent-harness.md) depth file.
**Depth files covered here:** currently none — this taxonomy tracks a set of tactics; individual depth files will land as specific strategies get published.

---

## The problem

Every agent step appends tokens to context: the last observation, the model's reasoning, the tool call, the tool's response. Left alone, coding-agent trajectories on SWE-Bench-scale tasks hit 200k+ tokens in tens of steps. This forces one of three failures:

1. **Overflow.** The next call exceeds the window and the agent crashes or silently drops the oldest turn.
2. **Cost blowup.** Each new step re-pays attention over the full history.
3. **Quality collapse.** Long-context degradation makes the model ignore critical earlier information.

A context manager exists to prevent (1)–(3) while retaining the parts of history the agent needs later.

## The shared pattern

Every variant does two things:

- **Classifies** each historical span as *keep verbatim*, *compress*, or *drop*.
- **Decides when to fire** — every step, at a token-budget threshold, or lazily on overflow.

They differ in *what does the classification* (rules vs. model), *how compression is done* (elision, summarization, or offloading to a store), and *whether dropped content is recoverable*.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Truncate-oldest | Drop turns from the front once budget hits | Loses task setup and early plans | Never, in practice — a baseline only |
| Rule-based elision | Deterministic patterns (strip stale tool output, keep last-N observations verbatim) | Fragile to unusual trajectories | Cheap default; sits before other layers |
| LLM summarization | Model rewrites older history into a condensed summary | Adds tokens/latency; loses fidelity | Very long horizons, single-agent settings |
| Staged rule → LLM | Rule elision first, then LLM summary only if still over budget | Slightly more machinery | **Modern default (Fan et al. 2026)** |
| Recoverable elision | Elided content is retrievable via a tool | Extra machinery; models rarely use it | Rare — most agents don't re-request |
| External memory / store | Long-lived off-context state (vector store, file scratchpad) | Requires the agent to explicitly query | Multi-session, cross-task recall |
| Delegated reading | Sub-agent reads a doc and returns a distilled note (SoL-Pi 2026) | Extra call; risk of loss | Reading a single very-long doc mid-trajectory |

## How to choose

Start with **staged rule → LLM** — it's the empirical winner for coding agents (Fan et al. 2026). Skip recoverable elision unless you can show your traces re-request elided content. Add an external memory store only once trajectories span multiple sessions or tasks. Delegated reading (SoL-Pi 2026) is worth adding when a single tool response can itself blow the window (a long doc, a long stack trace).

## Adjacent but distinct

- **KV cache compression** ([../inference/](../inference/)) — a *serving-side* trick that shrinks stored attention state. Complementary; both cut cost.
- **Attention windowing** ([../architectures/](../architectures/)) — a *model-side* structural trick (sliding window, linear attention). Different layer of the stack.

## Sources

- Paper: *An Empirical Study of Harness Design for Coding Agents* — Fan et al., 2026 — [arXiv:2609.20804](https://arxiv.org/abs/2609.20804).
- Paper: *SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness* — Liu et al., 2026 — [arXiv:2609.20519](https://arxiv.org/abs/2609.20519).
