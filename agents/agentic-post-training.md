# Agentic Post-Training Stages
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A class of RL post-training stages where the *rollout* is a full multi-turn agentic trajectory — tool calls, environment responses, plan updates — rather than a single-turn generation. Introduced as three back-to-back stages in the Rufus-Air recipe (**General Agent → Coding Agent → Search Agent**), all built on GRPO. Distinguishing features vs. single-turn RL: rollouts contain interleaved tool interactions with an environment; rewards blend terminal task success with turn-level rubric signals; sampling budget is dominated by environment latency rather than model forward passes.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/reward-reliability-ordering.md](../post-training/reward-reliability-ordering.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [rufus-air case study](../case-studies/rufus-air.md)

---

## What it is

Post-training an LLM into a competent agent requires RL over trajectories that *actually use tools*, not just prompts that mention tools. An agentic post-training stage wraps the base RL loop with an environment (sandbox, browser, code executor, retrieval index), rolls out complete multi-turn trajectories per prompt, and computes rewards on the trajectory as a whole. Rufus-Air's General/Coding/Search Agent stages are the reference implementation of this class.

## How it works

Each agentic stage plugs into a GRPO-style outer loop with three custom parts:

- **Environment.** A deterministic-enough sandbox: shell + files for a coding agent, browser or retrieval index for a search agent, general tool schema for a mixed-domain agent.
- **Trajectory rollout.** For each prompt, sample $G$ full trajectories in parallel. Each trajectory alternates model turns (reasoning + tool call) with environment turns (tool response). Trajectory length is bounded by a step limit.
- **Reward.**
  - *Coding Agent.* Repo-level task success from executed tests; optional turn-level penalty for pathological actions (infinite loops, over-long shell commands).
  - *Search Agent.* Final-answer correctness against ground truth; turn-level rubric on query quality and evidence use.
  - *General Agent.* Task-specific success signal per prompt; a mixed pool of tasks (small tool schemas, multi-step workflows, mixed domains).

Group-relative advantages, clipped-ratio update, KL to reference — all standard GRPO. The novelty is *what fills the trajectory*, not the algorithm.

## Why it matters

- **Concrete recipe for tool-use capability.** Rufus-Air's ordering (General → Coding → Search) is a template: broader capabilities first, then specialized loops.
- **Composes with single-turn RL.** Agentic stages assume the base model already has the reasoning and coding capabilities installed by tier-1 stages ([rlvr](../post-training/rlvr.md) on math + code). Skipping to agentic RL from base saturates the reward and stalls.
- **Different bottleneck.** Rollout throughput is set by environment latency (tool calls, code execution), not by GPU throughput. Infra choices (parallel sandboxes, cached shells) are load-bearing.

## Gotchas & tricks

- Environment determinism matters more than fidelity — flaky environments turn the reward into noise.
- Step-limit tuning is the difference between "solves the problem" and "trajectory times out on the last tool call"; set it above the median expert trajectory length.
- KL to reference must be tight or the agent's non-tool behavior (chat style, refusals) drifts by the end of the stage.
- These stages benefit from difficulty filtering ([rl-prompt-curation](../post-training/rl-prompt-curation.md)) as much as single-turn RL — tasks the base agent can't touch give zero signal, tasks it always solves are wasted rollout budget.
- Ordering across the agentic stages themselves matters. General Agent first builds broad tool literacy; specialized stages fine-tune it. Rufus-Air's ablations show swapping the order under-performs.

## Sources

- Paper: *Rufus-Air: An Open LLM Post-Training Recipe* — Amazon Rufus, 2026 — [arXiv:2609.29421](https://arxiv.org/abs/2609.29421). Introduces General / Coding / Search Agent as sequential stages within a longer pipeline.
- Related: *IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis* — Wu et al., 2026 — [arXiv:2609.29444](https://arxiv.org/abs/2609.29444). A specialized deep-search agent stage with role-decoupled policy optimization.
