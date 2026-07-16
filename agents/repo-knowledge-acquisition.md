# Repo Knowledge Acquisition (QA-Driven)
*Depth — pre-repair repository exploration that asks "what does the agent not yet know?" before it tries to fix.*

**TL;DR:** Coding agents fail on repo-scale issues because they don't understand enough of the codebase to make correct edits. Existing exploration methods are **fix-driven** — they follow the failing test's trail. ACQUIRE proposes **QA-driven** exploration: first generate questions the agent would need answered to make the fix, then explore the repo to answer those questions. This is a gap-first, not trail-first, retrieval strategy. Delivers up to **+4.4 Pass@1** on SWE-Bench-style issue resolution.

**Prereqs:** *(none)*
**Related:** [../evaluation/livecodebench](../evaluation/livecodebench.md), [../evaluation/humaneval](../evaluation/humaneval.md)

---

## What it is

A two-stage pre-repair framework for coding agents that solve real-world software issues (SWE-Bench-flavored). The framework interposes an **exploration stage** between issue ingestion and code editing:

1. **Question generation.** Given the issue and the reachable repo context, prompt the agent to enumerate concrete factual questions it would need answered before writing a patch. Examples: "Which module defines the abstract base class the failing test instantiates?", "What are the invariants callers assume about this function?"
2. **Answer retrieval.** Explore the repo (grep, file reads, call-graph traversal) targeted at those specific questions. The exploration budget is spent proportionally to how much a question would improve the agent's understanding.

The resolver stage then runs with the answered-question context prepended.

## How it works

The key claim is that the *right* retrieval signal for coding agents is not lexical similarity to the failing test (fix-driven) but the agent's own uncertainty about the repo (gap-driven). Concretely:

- **Fix-driven baseline.** Search for files whose names or symbols appear in the failing test, load them, hope the fix is in there. Fails when the fix is somewhere the test doesn't directly touch — e.g., a base class two hops away, a config default, a caller.
- **QA-driven ACQUIRE.** The agent enumerates *its own* knowledge gaps. Retrieval targets the gaps. Context is precise: it's what the agent needed to know, not what looked textually related.

On SWE-Bench-style benchmarks, this yields **up to +4.4 Pass@1** over the fix-driven baselines.

## Why it matters

- **Long-context retrieval is the ceiling on coding agents.** Model-side improvements (larger context, better attention) hit a wall when the agent doesn't know *what to retrieve*. QA-driven exploration is a retrieval-side improvement — orthogonal to model scale.
- **Fits how humans debug.** Real engineers ask questions ("where is this initialized?") before making changes. Making that explicit for the agent aligns retrieval with debugging.
- **Composable with any resolver.** ACQUIRE is a pre-processing stage; it drops in front of any downstream code-editing pipeline (LLM patch generation, agent scaffold, etc.).

## Gotchas & tricks

- **Question generation is the hard part.** If the agent's questions are vague ("what does this file do?"), retrieval returns everything and nothing. Questions must be pointed and answerable.
- **Budget must be finite.** Recursively spawning sub-questions blows up. The paper uses a fixed exploration budget with a scoring function to prioritize questions.
- **Complements, not replaces, model-side context.** Even with perfect QA-driven retrieval, the resolver still needs a competent code model. The gain is on top of whatever the underlying agent already does.
- **Distinct from ReAct-style reasoning.** ReAct interleaves questions with actions during the fix; ACQUIRE front-loads the questioning as a preparatory stage. In practice you'd combine them.

## Sources

- Paper: *Know Before Fix: QA-Driven Repository Knowledge Acquisition for Software Issue Resolution* — Lin et al., Shanghai Jiao Tong University, 2026 — arXiv 2607.11111.
