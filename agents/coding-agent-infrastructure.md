# Coding Agent Infrastructure
*Depth — the data layer beneath coding agents: repository indexes, symbol servers, per-commit views.*

**TL;DR:** Coding agents spend most of their turns discovering the same repository context — searching, reading files, resolving symbols — that a previous task already discovered. **Coding-agent infrastructure** is the data layer that lets those views be *built once per commit*, kept incrementally consistent across edits, and served through a stable per-agent runtime. The pattern generalizes: any long-lived agent working over a slowly-changing corpus benefits from a materialized multi-view data system rather than per-turn discovery.

**Prereqs:** [README](README.md)
**Related:** [agentic-search](agentic-search.md), [agent-memory](agent-memory.md)

---

## What it is

A per-repository (or per-corpus) service that maintains several *views* of the source, refreshes them against edits, and answers agent queries against them:

- **Lexical view.** Full-text / regex index (ripgrep, tantivy, ctags).
- **Dense view.** Embeddings of source ranges for semantic retrieval.
- **Structural view.** AST / LSP-derived symbol table for symbol-precise navigation.

Each result carries a **repository-relative source range** (file:start-end) the agent can cite verbatim. Views are keyed on a commit hash so that stale views can be detected and re-derived deterministically.

## How it works

```
on repo checkout at commit c:
    build_lexical(c), build_dense(c), build_structural(c)   # once
    store views keyed by c

on agent query q:
    view = pick_view(q)               # search? symbol? bounded-context?
    hits = view.query(q, budget)
    return hits with source ranges

on edit e:
    diff = e.diff()
    for view in views: view.refresh(diff)
    if refresh cost > threshold: rebuild from head commit
```

The agent's tool surface is a small, uniform API — `search`, `find_symbol`, `open_range`, `bounded_context(entity)` — hiding which view served the answer.

## Why it matters

- Amortizes indexing across tasks: for a repo touched by 100 agent sessions per day, per-commit views are ~free per session.
- Makes lifecycle costs *visible* (index storage, refresh compute) instead of hiding them inside opaque per-turn tool calls.
- Enables cross-task caching: if session B asks the same symbol lookup session A already resolved, the view returns instantly.
- Provides a natural place to enforce access control, PII redaction, and audit — one gateway, not many.

## Gotchas & tricks

- **Refresh cost dominates.** Rebuilding embeddings on every save is untenable; incremental dense-view updates require content-addressed chunking so unchanged code re-uses old vectors.
- **Structural view lags on broken code.** LSPs give bad results on files with syntax errors — a coding agent's edits mid-flight often *cause* those errors. Fall back to lexical view when structural fails.
- **View selection is a routing decision.** "Find the definition of `foo`" wants structural; "where do we handle rate limits?" wants dense; "grep for `TODO`" wants lexical. Wrong view is wrong answer.
- **Bounded context is a first-class query.** Agents don't want raw hits; they want a fixed-size context window packed with the most useful ranges. This is a ranking + budget problem, not a search problem.
- **Multi-repo generalization.** Real coding agents work across repos (main repo + dependencies). The infrastructure needs to federate views without leaking secrets across repo boundaries.
- **Trust the source range, not the snippet.** LLM tool responses often summarize; keep the raw source range so the agent can cite exactly.

## Sources

- Paper: *CodeNib: A Multi-View Data System for Serving Repository Context to Coding Agents* — Yu et al., 2026 — [arXiv:2607.25431](https://arxiv.org/abs/2607.25431).
- Related systems: Sourcegraph Cody, GitHub Copilot Workspace, Continue, aider — production coding-agent stacks with similar layering.
