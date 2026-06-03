# State-Externalizing Harness for Agent RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When you train an agent with RL on growing transcripts, the policy is forced to optimize *both* the semantic decisions (what to search, what to keep) *and* the routine bookkeeping (what's been seen, what's been verified, what's in budget). A **state-externalizing harness** moves bookkeeping into the environment — candidate pools, importance-tagged curated sets, evidence links, verification records, deduplicated observations, budget-aware context rendering — leaving the policy with only the semantic core. Harness-1 (UIUC, 2026) uses this partition to train a 20B search agent that hits 0.730 avg curated recall across 8 retrieval benchmarks, **+11.4 points over the next strongest open search sub-agent**.

**Prereqs:** [README](README.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/grpo.md](../post-training/grpo.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Search / browser / computer-use agents are typically trained as policies over growing transcripts: the model rolls out, the entire trajectory (queries, retrieved snippets, scratchpad notes) becomes the next observation, and RL optimizes the resulting policy. This conflates two very different jobs:

1. **Semantic decisions** — irreducibly model-shaped: what query to issue, which retrieved doc looks useful, when to stop.
2. **Bookkeeping** — *recoverable* from prior actions: what's already been seen, which evidence supports which claim, the running citation list, how many tokens of budget remain.

Bookkeeping doesn't need to be learned — the environment can maintain it more reliably and at zero learning cost. A *harness* is an environment wrapper that externalizes this state, presents it to the policy in a structured way, and only asks the policy to make semantic decisions on top.

---

## How it works

### The policy/harness split

```
                         ┌─────────── Environment (search APIs, web) ────────────┐
                         │                                                       │
   Policy π_θ            │     Harness (programmatic, no learning)              │
   ─────────             │     ──────────────────────────────────────           │
   - what to search ───▶ │  ─▶ issue query, fetch results                       │
                         │     update candidate pool                            │
                         │     dedup against prior observations                 │
   - what to keep   ◀───┤  ◀─ curated set with importance tags                  │
   - what to verify ───▶ │  ─▶ fetch full doc, record verification              │
   - when to stop   ───▶ │  ─▶ render budget-aware context summary              │
                         │                                                      │
                         └──────────────────────────────────────────────────────┘
```

The **harness state** in Harness-1: candidate pool, importance-tagged curated set, compact evidence links, verification records, compressed/deduplicated observations, and a budget-aware context renderer that summarizes the state into the policy's next prompt within a token budget.

The **policy's action space** shrinks to: query (semantic), keep/discard (judgment), verify (action), stop (termination).

### Training

Standard RL (GRPO-style, no value network) over outcome rewards (curated recall against held-out gold sets). Because the policy is no longer responsible for state management, the reward signal isn't diluted by bookkeeping mistakes — every gradient targets a semantic decision.

The harness itself is hand-engineered, not learned: dedup is by hash + semantic similarity; importance tags are heuristic; budget-aware rendering is a templating function. The contribution is the *partition*, not new ML.

---

## Why it matters

- **Disentangles policy from state.** RL excels at optimizing decisions under uncertainty; it's wasteful at memorizing dedup rules. Externalizing the recoverable state recovers the optimizer's focus.
- **Smaller models go further.** Harness-1's 20B policy is competitive with much larger frontier searchers (which carry the bookkeeping cost themselves). Same pattern likely applies to code agents and computer-use agents.
- **Reproducible and inspectable.** The harness is a program, not weights — auditable, deterministic, and easy to ablate (turn off dedup, see what breaks).
- **Aligns with broader "compute = harness × backbone" thesis.** Crafter (multi-agent figure generation, same digest day) makes a similar argument — wrapping a model with structured scaffolding beats scaling the backbone for tasks with a discrete component grammar.

---

## Gotchas & tricks

- **What counts as "recoverable" is task-specific.** For search: dedup, citations, verification are obviously recoverable. For research-style tasks with creative state (hypotheses, partial proofs), the line is fuzzier and over-aggressive externalization can strip useful intermediate scratchpad.
- **Budget-aware rendering is the trickiest piece.** Truncating the harness state for context-budget compliance can lose signal the policy needed. Importance tags help prioritize what to keep when rendering.
- **Eval against the right baseline.** Many "agent improvements" papers compare against unharnessed agents using the same scaffolding. Harness-1's +11.4 over the *next strongest open search sub-agent* is the apples-to-apples comparison.
- **Composable with [partial rollouts](../systems/partial-rollouts.md).** The harness state is checkpointable mid-rollout, enabling fault-tolerant long-horizon RL.
- **Doesn't eliminate the need for good prompts.** The policy still needs a clear schema for harness state in its context window — bad rendering means good harness state goes to waste.
- **Risk: harness encodes assumptions the policy can't override.** If the harness mis-tags importance, the policy may never see the dropped evidence. Worth giving the policy a "raw fetch" escape valve.

---

## Sources

- Paper: *Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses* — Jiang, Shi, Hong, Xu, Sun, Sun, Bashir, Han — UIUC, 2026 — [arXiv:2606.02373](https://arxiv.org/abs/2606.02373).
- Adjacent: tool-using-agent literature on context engineering (ReAct, Toolformer, ResearchGPT) — pre-Harness-1 work that informally relied on prompt templates for the same partition.
