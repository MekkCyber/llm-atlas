# Proactive Context Management
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pattern in which a long-horizon agent **actively curates its own working context** via specialized tools (plan, retrieve, summarize, offload to long-term memory, adaptively compress) instead of relying on heuristic eviction. When trained end-to-end, the RL loop assigns **action-level advantages** to the individual context edits rather than distributing one trajectory reward uniformly — so high-leverage edits get real credit.

**Prereqs:** [README.md](README.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

Standard long-horizon agents keep growing their working context until an eviction rule kicks in (recency, token cap). This wastes tokens on irrelevant history and loses information that later matters. Proactive context management treats *what to keep in context* as a learned decision, with dedicated tools and a training signal.

## How it works

**Enlarged toolset.** Beyond the usual search / delete / summarize primitives, the agent gets:

- **Planning tools** — externalize the current plan so it survives compression.
- **Long-term memory writes** — offload evidence out of the working window into a queryable store.
- **Soft context offloading** — move rarely-touched spans into a compressed representation the agent can still refer to.
- **Adaptive compression** — decide *when* and *how much* to compress.

**Fine-grained RL.** The training loop uses two ideas together:

1. **Critical-edit detection.** Watch **context variation** (how much the working context changed at each step) and **entropy variation** (how much the model's next-action distribution shifted). Steps with large joint variation are marked as critical decisions.
2. **Branch sampling around critical edits.** At a critical step, roll out several branches from the same state, each choosing a different context-editing action. Estimate an **action-level advantage** for each editing action from the outcomes of the branches that pass through it, rather than assigning the trajectory-level reward uniformly (as vanilla GRPO would).

## Why it matters

- Yields stronger long-context QA and deep-search performance while keeping a **more compact** working context — proving that eviction was the bottleneck, not raw window size.
- Consistent across base models and benchmarks, suggesting the training recipe transfers.
- Reframes agent memory as a learned control problem, not just a retrieval-store engineering problem.

## Gotchas & tricks

- Branch sampling multiplies rollout cost at critical steps — budget it explicitly. If you branch at every step, you're back to naïve rollouts.
- The "context variation" trigger is heuristic; if the compression tool is too coarse, every step looks critical and the credit signal collapses.
- Composes cleanly with GRPO — the branched, per-action advantages plug into the group-relative baseline without needing a value network.

## Sources

- Paper: *ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL* — Pan et al., Tencent, 2026 — [arxiv](https://arxiv.org/abs/2608.28476)
- Code: https://github.com/Tencent/ContextPilot
