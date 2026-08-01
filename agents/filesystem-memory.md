# Filesystem-Based Memory for LLM Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The de-facto long-term memory for deployed LLM agents is a **directory of markdown files** the agent reads, writes, and reorganizes through generic file tools (Claude Code, Cursor, Devin-style patterns). Zhou et al. (2026) run the first systematic study of that default: what organization actually buys, when it breaks, and how far the model choice matters vs the tool set. Headline: organization **halves retrieval cost** at scale, but no measured agent converts organization into *better* answers — only into cheaper ones — and organization *erodes* over time for all but the strongest management agent.

**Prereqs:** (none — this is the entry point for agent memory)
**Related:** none yet in this repo

---

## What it is

An "agent memory" that lives entirely as files in a directory tree. There is no separate database, embedding index, or bespoke schema — just markdown files and a shell / file-tool harness. The agent's own file operations *are* the memory API: `write` to remember, `read` to recall, `mv/mkdir` to reorganize.

The Zhou et al. paper formalizes filesystem memory around **three roles** operating on one store:

1. **Management agent** — integrates and organizes incoming content: writes new memories, resolves conflicts, promotes stale files, refactors the tree.
2. **Search agent** — answers queries by retrieving relevant files and citing sources.
3. **Execution agent** — supplies task trajectories that get distilled into reusable skill files, unifying declarative memory and skills in one store.

## How it works

The design space the paper varies (all in one framework):

| Axis | Values studied |
| --- | --- |
| **Memory shape** | agent-organized hierarchy · verbatim dump (append-only) · chunk retrieval |
| **Stream scale** | small → large (growth study) |
| **Tool harness** | sandboxed shell · memory-tool-style function set · varied search tooling |
| **Management strength** | weak → strongest available management LM |
| **Search strength** | weak → strongest available search LM |

Measured per config:

- **Answer quality** on long-conversation benchmarks and embodied tasks.
- **Cost** — total tokens and tool calls used per query.
- **Store health** — organization degradation over time (files/directory, duplication, staleness, orphaned entries).

## Why it matters

- **Filesystem memory is already the production default.** Named alternatives (RAG, vector DBs, bespoke agent-memory frameworks) fight against a rising tide of agents that just use `read_file` / `write_file`.
- **Measured findings challenge the folklore:**
  - Organized stores **roughly halve retrieval cost** where material is large — the reliable, reproducible win.
  - **Organization erodes** as memory grows for all but the strongest management agent tested.
  - **No agent measured converts organization into better answers** — only into cheaper answers.
  - The **tool set matters as much as the model** for the resulting store's shape.
- **Complements native-memory work.** [Metis](https://arxiv.org/abs/2607.26760) and [Memory Decoder at Scale](https://arxiv.org/abs/2607.27919) bet that memory should be parametric; this paper is the study of the external-memory alternative both are competing against.

## Gotchas & tricks

- **Organization ≠ quality.** If your only metric is answer accuracy, better organization won't show up. Track retrieval cost (tokens / calls / latency) too.
- **Store health drifts silently.** Duplication, stale files, and directory sprawl compound over many sessions — measure it explicitly, don't assume the agent will clean up.
- **Change the tool set and the store's shape changes as much as swapping the model.** A shell-only harness produces a very different tree than a curated `memory_add` / `memory_get` function set.
- **Chunk retrieval** (traditional RAG-style) is a valid fallback but sacrifices the "organize" affordance entirely — the paper puts it as a distinct memory shape, not a default.

## Sources

- Paper: *Filesystem-Based Memory for LLM Agents: Organization, Evolution, and Sustainability* — Zhou, Yu, Wei, Wu, Ouyang, Jiao, Pan, McAuley, Zhang, Yu, Han, 2026 — [arXiv:2607.26637](https://arxiv.org/abs/2607.26637)
- Contrast — parametric memory: *Metis: Memory Foundation Model* — Zhang et al., 2026 — [arXiv:2607.26760](https://arxiv.org/abs/2607.26760)
- Contrast — parametric memory at scale: *Memory Decoder at Scale* — Wei et al., 2026 — [arXiv:2607.27919](https://arxiv.org/abs/2607.27919)
