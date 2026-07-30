# Agentic Search
*Depth — LLM agents that explore corpora through tool calls instead of retrieving a fixed top-k.*

**TL;DR:** Classical retrieval-augmented systems pick top-k documents once and hand them to the LLM. **Agentic search** turns retrieval into an interactive loop: the LLM issues search / grep / follow-link queries, reads results, and decides what to search next, over many turns. This handles multi-hop questions that no single retrieval can satisfy, but pays a latency and correctness cost — the agent can wander, redo queries, or miss cheap evidence.

**Prereqs:** [README](README.md)
**Related:** [agent-memory](agent-memory.md), [coding-agent-infrastructure](coding-agent-infrastructure.md)

---

## What it is

A pattern with two subtypes:

- **Retrieval-augmented agents.** The tool is a semantic search endpoint returning ranked documents. The agent decides *what queries* to issue and *when to stop*.
- **Direct Corpus Interaction (DCI).** The tools are raw operators over the corpus — grep, glob, open-file, follow-reference — bypassing the retriever. The agent "reads" the corpus like a human developer reads a codebase.

DCI trades retrieval's relevance prior for exploration power: nothing narrows the space before the agent looks, so the agent can find evidence the retriever would have buried, but it also spends many turns groping around.

## How it works

Agent loop:

```
while not done:
    action = llm.plan(query, transcript)         # search, grep, read, or answer
    obs = tool.execute(action)
    transcript.append(action, obs)
```

DCI-specific tools are typically:

- `ripgrep pattern [path]` — regex search across files.
- `open path [lines]` — read a slice of a file.
- `list_symbols path` — structural navigation (functions, sections).
- `follow_reference symbol` — jump to definitions.

Retrieval-augmented variants use `search(query, k)` returning ranked snippets.

## Why it matters

- Multi-hop and browse-style questions that top-k retrieval can't answer (need a chain of lookups).
- Codebases and knowledge bases where structure matters — a symbol reference beats a semantic match.
- Frontier agent products (Perplexity, Deep Research, coding agents) all use variants of this pattern.

## Gotchas & tricks

- **Relevance-agnostic DCI wastes turns.** Pure grep-style search has no notion of "which file to open first." Layering a **relevance prior on top of interaction** — order documents by query relevance for sequential traversal, pick relevant entry paragraphs, rerank raw match snippets — improves the accuracy/efficiency frontier. See RARG ([arXiv:2607.24223](https://arxiv.org/abs/2607.24223)) for the concrete recipe.
- **Termination is a policy problem.** The agent needs a rule for "I have enough evidence, answer now." Without one, it either stops too early (weak answer) or spirals forever.
- **Grep matches are noisy.** A common failure mode is the agent believing the first grep hit is the answer without checking the surrounding context. Force `open` after `grep` before conditioning on a match.
- **Prompt injection through documents.** Corpus content is untrusted. Any instruction-like text the agent reads can hijack it. Treat tool outputs as data, not instructions.
- **Reward signals for training.** Search agents are often trained with RL over outcome correctness plus a step-count penalty; hyperparameters here decide whether the agent learns to be lazy or thorough.
- **Cache repeated queries.** In long sessions, agents re-issue near-identical queries. A small per-session cache is a free win.

## Sources

- Paper: *A New Role for Relevance: Guiding Corpus Interaction in Agentic Search* — Li et al., 2026 — [arXiv:2607.24223](https://arxiv.org/abs/2607.24223) — RARG: relevance-guided DCI.
- Paper: *Search-R1*, *WebGPT*, *Deep Research* system reports — earlier and concurrent examples of agentic search patterns.
