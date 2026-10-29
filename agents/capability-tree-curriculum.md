# Capability-Tree Curriculum
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Organize an agent's training not by task category (math, coding, embodied, …) but by the underlying *capability* each task actually exercises. Build a hierarchical capability tree, auto-generate simulated worlds that stress a target capability, train a specialized "Harness" (model + tools + prompt scaffolding) per capability, and route unseen tasks to the matching Harness at inference. The payoff is out-of-distribution transfer: capabilities compose across surface forms (an abstract reasoning puzzle and a navigation task can share a capability node).

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md) · [effort-escalation.md](effort-escalation.md)

---

## What it is

Standard agent-RL curricula group tasks by surface domain and sample uniformly or by difficulty. This produces specialist agents that transfer poorly: a math-trained agent can't navigate; a navigation-trained agent can't plan. Capability-tree curriculum answers the question "what *skill* does this task actually need?" and co-trains across tasks that share a skill.

Introduced in **The Trinity** (Tang et al., 2026) as part of a three-role recursive self-improvement loop: a **Creator** synthesizes focused simulated worlds, an **Explorer** runs the agent and collects trajectories, and an **Evolver** updates capability-scoped Harness configurations. A learned router selects the right Harness per task at inference.

## How it works

```
Capability tree (hierarchical):
      Spatial reasoning
      ├── Grid traversal
      ├── 3D navigation
      └── Diagram comprehension
      Multi-step planning
      ├── Tool-chain planning
      └── Constrained search
      ...
```

Training loop per capability node $c$:

1. **Creator** generates a batch of simulated worlds whose tasks require capability $c$ (and ideally little else).
2. **Explorer** runs the current Harness $H_c$ on these worlds and collects rollouts.
3. **Evolver** updates $H_c$ (model weights, tool configurations, prompts) using the rollouts, typically with on-policy RL.
4. Harnesses are *keyed* by capability node, not by surface domain. One Harness may cover an entire subtree; splits are triggered by capability-ambiguity signals from the Evolver.

Inference-time: a router $R(\text{task}) \to c$ picks the capability a task requires and dispatches to $H_c$. Multiple Harnesses can be composed if a task spans capabilities.

## Why it matters

- **OOD transfer.** Capability-sliced training generalizes to unseen tasks that share capabilities, including real-world embodied benchmarks. Trinity reports +34 pt mean gain across 28 training benchmarks and +26 pt across 18 unseen ones — a far wider gap than category-based curricula typically show.
- **Modular Harnesses.** Each Harness is small enough to iterate on. Replacing one capability's Harness doesn't disturb others.
- **Reframes RSI.** Recursive self-improvement becomes "evolve Harnesses against capability-targeted curricula" rather than "retrain the whole agent harder."

## Gotchas & tricks

- **Capability inventory is hard.** The tree is only as good as its decomposition. Trinity bootstraps the tree from the Creator's own analysis of generated worlds — a chicken-and-egg problem for a fresh domain.
- **World generation quality.** If the Creator's worlds don't actually isolate a capability (confound bleed-through), the Evolver trains the wrong signal. Trinity constrains the Creator with capability specs; still an empirical bottleneck.
- **Router accuracy caps everything.** A perfect Harness is useless if the router sends tasks to the wrong one. Harness coverage overlap gives graceful degradation.
- **Different from mixture-of-experts.** MoE routes *tokens* inside one model. This routes *tasks* across different model/tool configurations. The composition unit is coarser.

## Sources

- Paper: *The Trinity: Capability-Driven Recursive Self-Improvement for Multimodal Agents* — Tang et al., 2026 — [arXiv:2610.36077](https://arxiv.org/abs/2610.36077)
