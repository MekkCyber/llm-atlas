# Just-in-Time Memory (JitMem)
*Depth — deferring agent-memory curation from write time to read time.*

**TL;DR:** Most agent-memory systems distill each completed task's trajectory into a fixed artifact (reflection, workflow, skill) at write time, then retrieve it later by similarity. This forces you to decide what is worth remembering before knowing what the future query will be, and creates a long-horizon credit-assignment problem for training the curator. **JitMem keeps raw trajectories** and defers curation until read time — given the retrieved traces and the current task, a learned curator synthesizes a compact, task-adaptive payload. Because the payload is consumed on the same task, the curator learns from immediate task success. On ALFWorld / WebShop / τ²-bench it beats the best write-time-memory baseline by +16.2 / +16.3 / +3.9 absolute success-rate points.

**Prereqs:** [_rl](../post-training/_rl.md), [rlvr](../post-training/rlvr.md)
**Related:** [dual-track-memory](dual-track-memory.md)

---

## What it is

An agent-memory pattern that inverts the standard write-time distillation pipeline. Instead of committing to a summary the moment a task finishes, the system stores the raw trajectory (or lightly-annotated version) and a **curator** synthesizes a task-specific memory payload on demand when a new query arrives.

## How it works

Write path: complete a task → append the raw trajectory to a long-term store, along with lightweight metadata (task type, outcome, tokens). No summarization, no reflection, no skill extraction.

Read path (per new task):
1. **Retrieve** the top-k relevant trajectories by similarity.
2. **Curator** — a learned model — takes the current task + retrieved traces and emits a compact payload (relevant facts, past attempts, watch-outs) tailored to *this* query.
3. Agent solves the task using the payload as memory context.
4. Task success is the training signal for the curator (SFT on high-scoring payloads, or RL with success reward).

The key training-side benefit: the curator's reward is **immediate** (this task's success), not delayed across many future queries. This sidesteps the credit-assignment problem that plagues write-time memory learning — a stored artifact's value only becomes apparent when a matching query arrives many tasks later, potentially never.

## Why it matters

Reframes agent memory as a **retrieval + on-demand summarization** problem, not a "what to store" problem. This aligns with how modern RAG systems work and separates two axes that write-time memory conflates: information *retention* (keep everything raw) and information *presentation* (curate per query).

Empirical wins are large on task-diverse environments (ALFWorld, WebShop) where write-time summaries have to generalize across query types and inevitably lose useful specifics.

## Gotchas & tricks

- Storage cost is real — raw trajectories aren't free. Practical deployments will need trajectory compression that preserves information for retrieval (embedding indexes + delta compression) rather than semantic summarization.
- The retriever is the new weakest link: if it doesn't surface the right past traces, the curator can't rescue the situation. Retrieval quality dominates the curator's headroom.
- Training the curator directly on task success works when the payload's causal contribution to success is clear. Confounding by base-agent competence can be handled with contrastive rollouts (with/without payload).

## Sources

- Paper: *Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents* — Zhou et al., 2026 — [arXiv:2609.27334](https://arxiv.org/abs/2609.27334)
