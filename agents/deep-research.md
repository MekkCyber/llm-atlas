# Deep Research Agents
*Depth — the discovery-vs-verification asymmetry that structures long-horizon research agents.*

**TL;DR:** A "deep research" agent has to answer questions that jointly satisfy many constraints (multi-hop, multi-source, factual, quantitative). *Finding* such an answer is expensive; *verifying* a candidate factors into cheap constraint-wise checks. That asymmetry motivates a two-loop architecture — an inner research loop that gathers evidence and drafts an answer, and an outer verification loop that audits the answer constraint-by-constraint and re-launches targeted search for the parts that failed. Naive "search longer" pipelines conflate these; explicit verification is what turns extra compute into a durable answer.

**Prereqs:** [../agents/README](README.md), [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)
**Related:** [recursive-self-improvement](recursive-self-improvement.md), [../evaluation/README](../evaluation/README.md)

---

## What it is

Deep research is the class of tasks behind benchmarks like BrowseComp, WideSearch, DeepSearchQA, and Humanity's Last Exam: the answer is a synthesis of evidence found across sources under multiple binding constraints (people, dates, quantities, definitions). A single-shot ReAct loop rarely nails all constraints at once. Deep-research agents treat the answer as an *object that can be audited* rather than a token stream to complete.

## How it works

The canonical structure is nested:

```
outer_loop (verification):
    while unresolved_constraints:
        candidate = inner_loop(question, prior_state)
        audit = constraint_wise_check(candidate)
        unresolved = audit.failed_constraints
        prior_state = compress(prior_state, audit)   # keep verified + failed
```

Three moves make it work at scale:

1. **Constraint decomposition.** Split the question into checkable claims (a date, a person, a quantity, a definition) and score each independently. Verification is much cheaper than search.
2. **Targeted re-search.** Each failed constraint spawns a narrow sub-query rather than restarting the whole research session.
3. **Compressed state.** Long-horizon rollouts run out of context. A learned or scripted [context compressor](recursive-self-improvement.md) keeps verified evidence and unresolved constraints while dropping raw browsing traces.

Training pairs agentic mid-training on verified synthetic tasks with long-horizon RL on trajectories, giving denser credit to "decisive-evidence" steps (moments where a search resolved a constraint or reversed a wrong direction) to combat sparse final-answer rewards.

## Why it matters

Scaling deep research by rolling out longer traces plateaus quickly — each extra step is as likely to add noise as evidence. Scaling by *auditing more* extracts more signal from the same rollouts and composes naturally with tool-use RL. It's also the mechanism behind the current jump from single-hop QA agents to full research assistants.

## Gotchas & tricks

- **Auditors need calibrated skepticism.** If constraint checks are optimistic ("evidence looks reasonable"), the outer loop terminates on hallucinated answers. Prefer verifier prompts that default to *refuted*.
- **Compression is where budgets are won or lost.** External summarizers strip nuance; learned compressors trained end-to-end preserve exactly the artefacts the researcher will need next.
- **Beware verifier-search coupling.** If the same model both searches and verifies, it can produce evidence-agreeable-with-itself — split the search and audit prompts, or use a separate verifier model where possible.
- **Not every task needs it.** For single-fact lookups the two-loop overhead is pure waste; deep-research architectures shine when constraints number in the tens.

## Sources

- Paper: *AREX: Towards a Recursively Self-Improving Agent for Deep Research* — Lu et al., 2026 — [arXiv:2607.21461](https://arxiv.org/abs/2607.21461).
- Benchmarks referenced: BrowseComp, WideSearch, DeepSearchQA, Humanity's Last Exam.
