# OPD-Evolver — Slow-Fast Co-Evolving Memory Agent
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A memory-augmented agent recipe with a **fast loop** that interacts with a four-level memory hierarchy at test time and a **slow loop** that uses outcome-calibrated memory attribution and privileged hindsight to distill those memory-management skills into the deployable policy via on-policy self-distillation. A 9B model beats ReasoningBank-style memory systems by 11.5% and matches frontier-scale agents.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

"Self-evolving" memory agents (vector stores, scratchpads, retrieval-augmented planners) treat memory as a *storage* problem: persist trajectories, reflections, skills. OPD-Evolver argues the actual hard problem is *managing* memory — selecting which past experience to act on, writing back the right summaries, and maintaining a coherent repository over time — and that these skills should be *baked into the policy weights*, not bolted on as a retrieval system.

The framework is a **slow-fast co-evolution** loop: the fast loop uses external memory at inference; the slow loop watches what worked and trains the policy to do those memory operations from its own weights.

---

## How it works

### The four-level memory hierarchy

Memory is organized in four tiers with different read/write semantics:
1. **Episode buffer** — raw recent trajectories.
2. **Reflection store** — natural-language summaries of episodes.
3. **Skill library** — reusable procedures distilled from successful trajectories.
4. **Repository** — a curated long-term knowledge base.

The agent interacts with all four during the fast loop: it **reads** from each, **uses** the retrieved content to act, **writes** back outcomes, and **maintains** the higher tiers by curating from the lower ones.

### Fast loop — test-time evolution

For each task, the agent:
1. Reads from the four-level hierarchy.
2. Plans / acts using the retrieved content.
3. Records the outcome.
4. Writes reflections / skills / repository entries based on what helped.

This is the "evolver" behavior — memory grows and reorganizes between tasks.

### Slow loop — outcome-calibrated memory attribution

After episodes complete, the slow loop has *privileged hindsight* — it knows which memory operations the fast-loop policy made and whether the outcome was good. The attribution step decides which memory reads / writes actually contributed to success.

The slow loop then runs **on-policy self-distillation** on the policy: the teacher is the policy plus the privileged-hindsight-annotated trajectory; the student is the policy without hindsight. Over iterations, the student internalizes the memory selection / writing skills the teacher gets from hindsight — pushing the four abilities (read, use, write, maintain) from the external memory system into the weights.

### The four learnable abilities

The slow-loop distillation explicitly targets:
- **Select** the right past experience.
- **Act** on it correctly.
- **Write** reusable knowledge back.
- **Maintain** the repository (consolidation, eviction).

Each ability has its own attribution signal in the slow loop.

---

## Why it matters

- **Compresses memory-augmented systems into the policy.** ReasoningBank-style systems carry heavy retrieval / memory subsystems at inference. OPD-Evolver pushes the policy to do the same work from its own weights, dropping deployment complexity.
- **9B model challenges much larger agents.** OPD-Evolver-9B beats ReasoningBank by up to 11.5% and training-based methods (Skill0) by ~5.8%, and challenges Qwen3.5-397B-A17B / Step-3.5-Flash. Memory internalization is doing real work, not just bookkeeping.
- **Generalizes the on-policy distillation pattern.** Slow-fast co-evolution is a clean framework: anywhere there's "external scaffolding the policy uses at test time" (memory, tools, retrieval), the same hindsight-distillation pattern can push that capability into weights.

---

## Gotchas & tricks

- **Attribution is the hard part.** Naively distilling all memory reads loses signal; the hindsight has to decide which were causally helpful. Paper uses outcome-conditioned credit assignment over the four tiers.
- **Memory hierarchy is not load-balanced automatically.** Repository inflation is a real failure mode; the *maintain* ability includes eviction and consolidation, which the slow loop explicitly trains for.
- **Slow loop is expensive.** Even though it's outcome-calibrated, it does run rollouts with privileged context. Treat it as periodic post-training rather than continuous.
- **Doesn't replace tools.** The model still benefits from real-world tool calls; OPD-Evolver compresses the *memory* layer, not the broader agent loop.

---

## Sources

- Paper: *OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation* — Guibin Zhang et al., NUS, 2026 — [arXiv:2606.17628](https://arxiv.org/abs/2606.17628).
