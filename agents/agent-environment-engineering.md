# Agent Environment Engineering
*Depth — designing the agent's environment along four axes (permissions, artifacts, budgets, oversight) instead of micromanaging its workflow.*

**TL;DR:** As base agents get more capable, the bottleneck shifts from prescribing workflows to **designing the environment** in which the agent acts. EurekAgent (2026) formalizes this as a four-axis framework — *permissions*, *artifacts*, *budgets*, *human oversight* — and reports state-of-the-art results on mathematics, GPU-kernel engineering, and ML tasks, including circle-packing discoveries for under $11 of API cost.

**Prereqs:** *(none)*
**Related:** [README.md](README.md) · [code-as-action.md](code-as-action.md)

---

## What it is

A design framework for autonomous agents where the developer's main responsibility is not crafting the prompt/plan but defining the agent's operating environment. Four axes:

- **Permissions** — what the agent can read, write, execute, observe.
- **Artifacts** — what state persists across turns (files, variables, scratchpad, results database).
- **Budgets** — compute, tool-call, and dollar caps that bound exploration.
- **Human oversight** — checkpoints, approval gates, intervention surfaces.

The agent's reasoning loop and prompt stay generic; the environment is what varies between deployments.

---

## How it works

### Permissions

Define a capability surface explicitly: file paths, tool subsets, network reach, side-effect boundaries. Capability is what the agent *can* do; the framework asks the developer to specify this upfront rather than discover it through agent misbehavior.

### Artifacts

Decide what survives between turns. Code agents persist the codebase + a scratchpad; research agents persist a results database. Artifacts are the agent's long-term memory and the developer's window into progress.

### Budgets

Compute (tokens, GPU-seconds), tool calls, and dollar limits per task. Budgets are the *exploration limiter* — they let the agent search broadly without unbounded cost.

### Human oversight

Where humans intervene: pre-execution review for risky actions, post-execution audit, periodic checkpoints. Oversight is a budget-priced operation, so it competes with agent exploration in the design.

### Empirical recipe

EurekAgent's reported wins (mathematics, kernel engineering, ML) come from holding the agent fixed and varying the environment. The framework is *what to design*, not a specific implementation.

---

## Why it matters

- **Where agent leverage is moving.** With capable base models, the marginal value of fancier orchestration plateaus; environment design becomes the differentiator.
- **Makes agent behavior reproducible.** Specifying permissions/artifacts/budgets upfront pins the agent's surface; otherwise reproducibility depends on undocumented prompts.
- **Cost-controlled discovery.** Circle-packing discoveries for ~$11 of API spend is a vivid demonstration of well-designed budgets.

---

## Gotchas & tricks

- **Permissions creep.** A permissive default ("anything in this directory") drifts into unintended side effects. Explicit allowlists work better than denylists.
- **Artifact bloat.** Persistent state grows; need a compaction or retention policy or the artifact store becomes the bottleneck.
- **Budgets need feedback.** An agent that doesn't see its remaining budget can't allocate effort sensibly; surface budget state in each turn.
- **Oversight cadence matters.** Too-frequent checkpoints kill autonomy; too-rare misses errors. Tie checkpoint frequency to artifact-write frequency, not turn count.

---

## Sources

- Paper: *EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery* — Xin et al., Tsinghua + Zhipu AI, 2026 — [arXiv:2606.13662](https://arxiv.org/abs/2606.13662).
