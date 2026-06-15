# Shortcut-Resistant Task Synthesis

*Depth — synthesizing agent training tasks whose intended difficulty isn't bypassable by a cheap identifying route.*

**TL;DR:** Synthetic search-agent training data is often "fake hard": the graph topology looks complex but a shortcut identifying route trivializes the task — the agent learns to find the cheap path instead of actually searching. **FORT** is a task-synthesis framework that explicitly probes for shortcut risk at multiple synthesis stages (entity choice, evidence linking, query phrasing) and discards anything a cheap retriever can solve. **FORT-Searcher** trained on FORT data is competitive with (and beats) RL-trained baselines using **SFT alone**.

**Prereqs:** [_data-curation.md](_data-curation.md)
**Related:** [decontamination.md](decontamination.md), [quality-filtering.md](quality-filtering.md), [../agents/README.md](../agents/README.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

A data-curation principle for *training* deep search agents. Existing synthesis methods raise apparent difficulty by enriching graph structures (more entities, more hops, richer relations). But topological complexity is not the same as *realized* search difficulty: a complex question with a unique high-frequency identifier still collapses to a one-step retrieval.

Shortcut-resistant synthesis measures difficulty by *what a cheap solver actually has to do*, not by graph topology.

## How it works

### Stages where shortcuts emerge

| Stage | Shortcut risk |
| --- | --- |
| Entity selection | If the gold entity is uniquely identified by a rare surface form, a one-shot retrieval wins |
| Evidence linking | If the entity is reachable through one popular link, multi-hop topology is irrelevant |
| Query phrasing | If the query contains a distinctive phrase, embedding retrieval bypasses search |

### The FORT probe

At each stage of synthesis, a cheap-solver probe attempts to short-circuit the intended search path:

1. **Surface-form probe.** Try retrieving the gold entity by the most distinctive phrase in the question. If it succeeds, the question is rephrased or rejected.
2. **One-hop probe.** Try a single retrieval from the question to the gold answer. If it succeeds, the multi-hop structure is illusory.
3. **Embedding probe.** Try dense retrieval of the gold passage from the question. If it succeeds, no search-style reasoning is needed.

Tasks that any probe solves are discarded or modified. Only tasks where the intended search depth is *realized* survive.

### Training outcome

FORT-Searcher is fine-tuned on FORT-synthesized data with vanilla SFT. On challenging deep-search benchmarks it matches or exceeds prior baselines that combined SFT *with* RL — the data quality dominates the training paradigm.

## Why it matters

- **Data quality > training paradigm for agents.** The "is your benchmark actually hard?" critique applied to training data, with the headline result that careful synthesis lets SFT-only training beat SFT+RL.
- **Transferable principle.** The shortcut-probe pattern generalizes to other agent task families (tool-use synthesis, code-agent task generation): name the cheap solver, probe with it, discard solvable cases.
- **Removes a hidden reward-hacking surface.** Agents trained on shortcut-rich data learn shortcut policies, which then fail in deployment when the shortcuts are absent.
- **Cheaper than RL.** Investing in data curation up front trades against expensive RL rollouts downstream.

## Gotchas & tricks

- **The probes define what "shortcut" means.** Probes must be the cheapest solvers — if you only probe with one weak retriever, more powerful retrievers will still shortcut. Use the strongest cheap solver as the probe.
- **Aggressive filtering shrinks data.** Many synthesized tasks die at the probes; budget for the throughput loss.
- **Surface-form rewriting is hard.** Rephrasing to defeat surface-form probes can introduce ambiguity. Easier to discard than to rewrite.
- **Verify post-hoc on a held-out cheap-solver baseline.** Even after the probes, a stronger cheap solver might re-discover shortcuts. Periodic audits matter.
- **Doesn't replace decontamination.** Train-time shortcut filtering and test-time contamination filtering ([decontamination.md](decontamination.md)) are independent concerns.

## Sources

- Paper: *Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents (FORT-Searcher)* — Chen, Xiang, Zeng et al., KAUST / Renmin / SJTU, 2026 — [arXiv:2606.12087](https://arxiv.org/abs/2606.12087).
- Related: [_data-curation.md](_data-curation.md), [quality-filtering.md](quality-filtering.md).
