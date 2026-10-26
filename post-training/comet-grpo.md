# COMET-GRPO

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant that **splits the advantage signal** for a policy that makes two nested choices (which *mode* of execution, and which action within that mode). Each decision gets its own advantage, so credit for the outer choice doesn't contaminate the inner one and vice-versa. Introduced with RouteWeaver, where the outer choice is execution style (single call / sequential / multi-agent) and the inner choice is model-and-tool selection within that style.

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md)
**Related:** [rlvr](rlvr.md)

---

## What it is

Standard GRPO (and PPO) treats a trajectory as a single sequence of actions with one advantage per action (or one advantage per response in group-wise variants). When the policy actually makes *nested* decisions — pick a mode first, then pick an action within the mode — treating these as a flat action sequence collapses the credit assignment: a bad final reward can be attributed to the mode choice, the within-mode action, or both.

COMET-GRPO (Compositional Mode-Execution Training with GRPO) introduces a per-decision advantage. The outer (mode) decision gets its own group-normalized advantage; the inner (execution) decisions get their own. Both decisions update the same policy, but they're updated with signals that don't collide.

---

## How it works

### Nested decisions

For each query, the policy produces:

1. A **mode** $m \in \mathcal{M}$ (e.g. {single-call, sequential-call, multi-agent}).
2. A **sequence of execution actions** $a_1, a_2, \ldots$ *conditioned on* $m$ (e.g. which model to call, which tool).

The trajectory ends with a scalar reward $r$ (task success).

### Two advantage streams

**Group-normalized mode advantage.** Within a group of $N$ sampled rollouts for the same query, compute per-mode returns and normalize across the group:

$$
A^{(m)}_i = \frac{r_i - \bar{r}_{\text{group}}}{\sigma_{\text{group}}}
$$

This advantage updates the mode-selection head only.

**Group-normalized execution advantage.** Within rollouts that chose the same mode $m$, compute within-mode advantages:

$$
A^{(a)}_i = \frac{r_i - \bar{r}_{m}}{\sigma_m}
$$

This advantage updates the within-mode action tokens only.

Updates are summed:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{PG}}(A^{(m)}, \text{mode tokens}) + \mathcal{L}_{\mathrm{PG}}(A^{(a)}, \text{execution tokens}) + \mathrm{KL}
$$

### Progressive exploration

A separate ingredient from the signal split: start RL with the mode decision **assigned** externally (not from the policy), so the policy learns strong within-mode execution first. Gradually anneal to **autonomous** mode selection. Prevents collapse where the policy locks onto one mode before it has learned to execute well in any.

---

## Why it matters

- **Clean handling of nested policies.** Any agentic policy that chooses "what kind of thing to do" and then "how to do it" has this nested structure. COMET-GRPO's split is the clean fix.
- **Load-bearing per the paper's ablations.** RouteWeaver loses most of its +11% advantage over the best baseline when the advantage split is removed, and collapses to single-round execution when progressive exploration is removed.
- **Compatible with standard GRPO infra.** The only change from vanilla GRPO is masking advantages per token category; no new sampling or scoring machinery is needed.

---

## Gotchas & tricks

- **The split matters only when both decisions have real variance.** If one decision is deterministic (e.g. mode is always "sequential"), COMET-GRPO degenerates to standard GRPO on the other decision. The gain shows up when both decisions genuinely matter.
- **Token-attribution to decisions.** The split assumes you can label which output tokens belong to the mode decision vs the execution decision. For free-form generation this requires a structured output format (e.g. a mode tag at the start).
- **Progressive exploration is a schedule, not a term.** It's not a loss change — it's a curriculum. Needs the environment to accept externally-assigned modes during warmup.
- **Generalizes beyond routing.** The same split applies to any agent that picks a plan type then executes it (tool-choice-then-tool-use, policy-type-then-policy, etc.).

---

## Sources

- Paper: *RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing* — Wang, Liu, Feng, Wang, NTU / UIUC / HUST, 2026 — [arXiv:2610.19705](https://arxiv.org/abs/2610.19705).
- Precedent: *GRPO* — introduced in DeepSeekMath / DeepSeek-R1, 2024-25 — the group-normalized-advantage policy-gradient variant COMET-GRPO extends.
