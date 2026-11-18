# Multi-Agent LLM Systems

*Taxonomy — architectures where multiple LLM calls coordinate, ensemble, or deliberate on the same task.*

**TL;DR:** "Multi-agent" is an umbrella for three primitives that often get conflated: **coordination** (agents talk to each other), **ensembling** (multiple samples pooled by voting or selection), and **test-time compute** (more thinking or steps per call). Across a 2026 meta-analysis of 516 configurations, most reported multi-agent gains disappear once baseline compute is matched: coordination without diverse evidence or external verification adds no information, so a single-agent baseline with the same tokens beats it on average by ~0.042 accuracy. Multi-agent wins only when you deliberately introduce *information asymmetry* (partitioned evidence), *verification* (an external oracle or executable check), or *model heterogeneity*.

**Related taxonomies:** [../post-training/_rl.md](../post-training/_rl.md)
**Depth files covered here:** (none yet)

---

## The problem

Multi-agent LLM pipelines are everywhere — debate, planner-executor, critic loops, swarms, role-play panels — and they usually *look* better than a single call. The question the meta-literature answers is: better than *what*? Most published comparisons hold the per-call compute fixed, so "N agents" burns N times the tokens of the single-call baseline. Once you match total tokens or total FLOPs, the picture flips.

A clean taxonomy helps: it separates what the design is *actually* contributing (more thinking, more samples, or genuinely new information from other agents) from the compute scale-up, and tells you which multi-agent setups could in principle win.

---

## The shared pattern

Every multi-agent design is built from some combination of three primitives:

```
coordination   : agent_i reads agent_j's output and updates its own
ensembling     : K independent samples pooled by vote / selection / weight
test-time      : more tokens or reasoning steps per call (CoT, loops, search)
```

The reason the compute-matched comparison matters is that *ensembling* and *test-time compute* are already strong, well-studied baselines. "N agents in a debate" has to beat (a) N independent samples from the same model with majority voting, and (b) 1 sample from the same model with N× the reasoning budget — both of which cost the same.

---

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Debate / discussion | agents exchange answers across rounds, converging | mostly redundant samples; often no evidence delta | heterogeneous model pool with diverse priors |
| Planner-executor | one agent decomposes, another executes | needs executor to be *better* at the subtask | subtasks are structurally easier than whole task |
| Critic / reflection | one agent reviews another's output | critic suffers same blind spots | external verifier (tests, oracle) replaces LLM critic |
| Role-play panels | agents with assigned personas discuss | personas rarely give evidence the base model lacks | partitioned evidence actually assigned per role |
| Majority voting / self-consistency | K independent samples, mode wins | one of the strongest baselines; purely ensembling | any reasoning task with sampleable diversity |
| Mixture-of-experts at inference | route each query to a specialist | needs real specialists, not re-prompted general models | truly heterogeneous experts |

---

## How to choose

Default: start from a single-agent baseline with CoT or an appropriate test-time compute budget. Only move to multi-agent when **one of three conditions holds**:

- **Partitioned evidence.** Different agents have access to different data (one has the document, another has the database, a third has code execution). Communication now transmits real information.
- **External verification.** A non-LLM verifier — unit tests, a solver, a sandbox — grades each candidate. The multi-agent pipeline is now ensembling against a reliable filter.
- **Model heterogeneity.** Agents are genuinely different models (different training data, sizes, families), not the same model with different prompts.

Without any of these, use sampling + voting at the same compute; the meta-analysis shows it wins on average. For stress-testing reasoning (math, code) with executable grading, use rejection sampling against the verifier rather than an agent panel.

## Adjacent but distinct

- **Self-consistency / majority voting** — pure ensembling, not multi-agent. Belongs on the single-agent + TTC axis.
- **Agent frameworks (MCP, tool use)** — about *what* one agent can do, not *how many* agents interact. See [../agents/README.md](../agents/README.md).
- **RLHF-style debate (Irving et al.)** — training-time debate as an oversight mechanism. Shares the word "debate" but the goal is alignment, not accuracy per task.

---

## Sources

- Paper: *More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs* — Sajjadi Mohammadabadi, U. Nevada Reno, 2026 — [arXiv:2611.17610](https://arxiv.org/abs/2611.17610) — the 516-configuration meta-analysis this taxonomy is built on.
- Paper: *Self-Consistency Improves Chain of Thought Reasoning in Language Models* — Wang et al., 2022 — the sampling-plus-voting baseline that a lot of multi-agent work fails to beat.
- Paper: *Improving Factuality and Reasoning via Multi-Agent Debate* — Du et al., 2023 — the canonical multi-agent-debate result.

---

## Conventions

- **Filename:** `_multi-agent.md` (leading underscore — taxonomy).
- **Folder placement:** `agents/`.
- **Scope:** architectures with >1 LLM call per task. Single-call test-time compute (CoT, sampling) belongs in [../post-training/reasoning/](../post-training/reasoning/).
