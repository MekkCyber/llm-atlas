# Span-level agent-trajectory evaluation (TELBench / DRIFT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Most agent benchmarks score only the final answer or holistically label a whole trajectory. TELBench (Wang et al., 2026) introduces *span-level error localization*: given an agent's trajectory, the model must point to the contiguous semantic span where a downstream-harmful error was introduced. The companion DRIFT framework audits each *claim* the agent forms — introduction, support, dependency — to separate genuine errors from benign exploration.

**Prereqs:** [README.md](README.md), [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)
**Related:** [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md), [../agents/README.md](../agents/README.md)

---

## What it is

Deep-research agents produce long trajectories — alternating between searching, reading, reasoning, and synthesizing — across many tool calls. Existing evaluations either:

- score only the final answer (ORM-style) — gives no signal about *where* the agent went wrong, and rewards "look right at the end" over "be right throughout";
- holistically label the trajectory ("good" / "bad") — averages over both real errors and recoverable exploration.

Span-level evaluation asks: can the model (or, more importantly, *a different judge*) point to the *span* of the trajectory where a harmful error was introduced? This is the agent-trajectory analogue of a process reward.

## How it works

1. **Trajectory decomposition.** Split the trajectory into ordered semantic spans — each span is a coherent reasoning, search, or synthesis step.
2. **Annotation.** Human experts label each span with one of: benign exploration, recoverable error, *harmful* error (error that propagates downstream and is depended on by later spans).
3. **TELBench dataset.** 2,790 annotated trajectories, used to evaluate whether a model can recover the labeled spans from the trajectory alone.
4. **DRIFT auditing.** Tracks each *claim* the agent makes — introduction, the evidence cited to support it, and which later steps depend on it. A span is harmful when it introduces a claim that (a) fails support verification *and* (b) is depended on by later spans.
5. **Evaluation metric.** Span-level F1 / IoU against expert labels.

The split between *recoverable* and *harmful* errors is the key conceptual move — exploration that the agent self-corrects is benign; a wrong claim that the agent keeps building on is the failure to detect.

## Why it matters

- **Process signal for agents.** Span-level labels are the data substrate for training process reward models or step-level critics on agent trajectories — the agentic equivalent of PRMs in reasoning RL.
- **Hides-failure agent paradox.** Final-answer scoring rewards agents that *recover* visibly bad reasoning. Span-level scoring penalizes the underlying mistakes. This tilts the optimization target toward genuinely correct intermediate reasoning.
- **Claim graphs are a reusable primitive.** DRIFT's introduce → support → dependency triple is broadly applicable to any agent that makes claims — research, coding, planning.

## Gotchas & tricks

- **Annotation cost.** Span-level labels are far more expensive than outcome labels. Synthetic annotation (LLM-as-judge over trajectories) is a possible bridge but inherits its own biases.
- **Granularity tradeoff.** Spans too small → noisy labels; spans too large → harmful and benign content mixed.
- **Recovery counts.** Agents that explore wrong directions but recover should not be penalized. DRIFT's support-verification step is the key gate.
- **Trajectory diversity.** The 2,790-trajectory corpus skews toward research-style tasks; coding and tool-heavy agent trajectories may need their own corpora.

## Sources

- Paper: *Where Do Deep-Research Agents Go Wrong? Span-Level Error Localization in Agent Trajectories* — Wang et al., 2026 — [arXiv:2606.02060](https://arxiv.org/abs/2606.02060).
- Related: PRM (process reward models) literature; trajectory-level RL critics.
