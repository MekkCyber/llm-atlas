# History-Guided Trajectory Collection
*Depth — picking where to branch an agent's rollouts based on the history of prior attempts on the same problem.*

**TL;DR:** For long-horizon agent tasks (SWE-bench-style coding, computer-use), naive repeated sampling wastes compute on near-duplicate trajectories. **SWE-Trail** (Ant Group, 2026) uses the *history of previous attempts on the same issue* to pick high-value branching states, then at each such state proposes multiple candidate actions, briefly executes each from the same checkpoint, and keeps the branch whose repository evidence (tests, type checks) wins. The agent itself never sees the history — it only benefits from smarter *where to explore*. Training a 124B student on SWE-Trail trajectories gives +2.33 points on SWE-bench Multilingual and +1.60 points on SWE-bench Pro over independent sampling.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md) · [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Trajectory collection for agent training has two layers:
- **Inner policy** — the agent itself, deciding what to do at each state.
- **Outer collection strategy** — the training-time mechanism that decides which rollouts to run and keep.

Default outer strategy is **independent sampling**: for each problem, run `N` independent rollouts, keep whichever pass tests, train on them. Near-duplicate rollouts are frequent; compute is wasted.

**History-guided trajectory collection** replaces the outer strategy with a search over the state graph, biased by what previous runs discovered. The inner agent still runs normally and sees no history; only the outer collector consults it.

## How it works

Per issue, maintain a history: the sequence of decisions each previous run took and whether each run passed. On a new collection pass:

1. **Pick branching states.** Use the history to identify states where previous runs diverged or reliably failed. These are the high-value places to explore more.
2. **Branch.** At each branching state, run the agent `k` steps forward with `k` different sampled actions, each from the same repo checkpoint.
3. **Score branches by repo evidence.** The agent itself doesn't see test results until the end of a rollout, but the collector can run partial checks — build, type-check, lint, targeted tests — and keep the branch whose evidence looks best.
4. **Continue.** Only the kept branch extends; the others are discarded.

The agent is thus doing beam-search-with-oracle-selection, but the oracle is *the repository*, not an external LLM. SWE-Trail reports this converts naive-sampling trajectories into trajectories that look like a smart human's — branching on unclear choices and keeping only the paths that build.

## Why it matters

- **Trajectory-quality bottleneck.** Coding-agent RL is bottlenecked less by algorithm than by trajectory quality; naive sampling plateaus at near-duplicate solutions. History-guided exploration opens the state graph instead of the decoding distribution.
- **No reward hacking surface added.** Because the branching score uses repository mechanics (tests, builds) rather than an LLM judge, there's no judge-reward to hack.
- **Transferable framing.** The same inner-policy / outer-collector split applies to other agent domains — computer use, browser agents, scientific workflows — wherever the outer world has cheap partial-outcome signals.

## Gotchas & tricks

- **Branching cost explodes.** `k × depth` extra forward passes per issue. Pay attention to how deep the branching goes and how many problems you're processing.
- **Partial-check selection bias.** If the partial checks don't correlate with final success, branch selection follows the wrong signal. Validate on held-out issues.
- **History locality.** History from *different* issues isn't useful; keep history indexed by issue (or by repository state) rather than global.
- **Agent must not see history.** If the agent sees the history, you're just doing in-context learning, not state-graph search.

## Sources

- Paper: *SWE-Trail: History-Guided Trajectory Collection for Coding Agent Training* — Lu Yu, Jiatan Huang et al., Ant Group / U. Connecticut, 2026 — [arXiv:2611.02074](https://arxiv.org/abs/2611.02074).
- Benchmark: SWE-bench Multilingual, SWE-bench Pro — training and evaluation harness.
