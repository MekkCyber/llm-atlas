# Role-Typed Credit Assignment (TRIAGE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard agentic RL broadcasts one outcome-level advantage across every action token in a trajectory. This fails to credit useful *exploration* inside a losing trajectory and reinforces *regressive* actions inside a winning one. TRIAGE (Xu et al., 2026) classifies each trajectory segment into one of four semantic roles — **decisive progress**, **useful exploration**, **no-progress infrastructure**, **regression** — and applies role-specific credit rules on top of the outcome signal. The paper proves this role-conditioned correction is optimal given only role labels.

**Prereqs:** [../grpo](../grpo.md), [../_rl](../_rl.md)
**Related:** [prm](prm.md) · [long-cot-rl](long-cot-rl.md) · [../_rewards](../_rewards.md) · [../../agents/README](../../agents/README.md)

---

## What it is

Agentic RL trajectories are long and noisy: dozens of tool calls, retries, dead ends. GRPO-family algorithms give every token in the trajectory the same scalar advantage $A = (r - \bar r) / \sigma_r$. Two failure modes:

- **Positive-outcome trajectory.** All tokens get positive credit, including *regressions* (backtracking, wrong queries) that dragged the trajectory sideways. RL reinforces them.
- **Negative-outcome trajectory.** All tokens get negative credit, including *useful exploration* (probing that ruled out a hypothesis). RL punishes them.

TRIAGE fixes both by adding a **role axis** to credit assignment: not just "was the trajectory good?" but "was this segment's role good given the outcome?"

## How it works

### The four roles

Every segment (typically one tool call or one thought-action chunk) is classified into one of:

- **Decisive progress** — moves the trajectory materially toward the goal.
- **Useful exploration** — probes / rules out; doesn't directly progress but informs later decisions.
- **No-progress infrastructure** — bookkeeping, formatting, non-substantive.
- **Regression** — undoes prior progress, or takes a step known to be wrong.

Role labels come from a lightweight classifier (a small LLM fine-tuned on annotated trajectories, or a rule-based judge on structured actions).

### Role-conditioned credit rules

Given the outcome-based advantage $A$ and a segment role $\rho$, TRIAGE applies bounded segment-level process rewards:

$$
A^{\text{TRIAGE}}_{s,t} = A + \delta(\rho_s)
$$

where $\delta$ is a bounded role-specific bonus/penalty:

- decisive progress → $+\delta^+$
- useful exploration → $+\delta_{\text{exp}}^+$ **even if trajectory failed**
- no-progress → $\approx 0$
- regression → $-\delta^-$ **even if trajectory succeeded**

The bounds keep the outcome signal as the primary optimization direction; process rewards are corrections, not overrides.

### Optimality result

The paper proves that role-conditioned credit is the *optimal* correction expressible from role labels alone: any finer correction requires strictly more information than the role tag.

## Why it matters

- **Consistent gains over GRPO** on ALFWorld, Search-QA, and WebShop, with additional trajectory-length efficiency.
- **No PRM required.** Unlike full process-reward models (PRMs), TRIAGE only needs a segment-role classifier — much cheaper and more sample-efficient to build.
- **Plug-and-play.** Drops into any GRPO-family recipe (RLVR, long-CoT RL) with a minimal change to advantage computation.
- **Interpretable credit.** Debugging why a policy learned a strange behavior is easier when the credit signal has semantic labels.

## Gotchas & tricks

- **Role classifier quality is the floor.** A noisy classifier injects noisy credit; validate its agreement with human labels before deploying.
- **Class imbalance.** "No-progress" dominates most trajectories; sample or weight the classifier training to avoid collapse to the majority class.
- **$\delta$ magnitudes are hyperparameters.** Keep them bounded relative to the outcome advantage — otherwise the process reward drowns the outcome reward and TRIAGE decays into a pure process-reward method.
- **Task-specific role taxonomies.** The four-role scheme fits web / search / interactive tasks; math reasoning may want a different partition (algebraic move / verification / restatement / mistake). Redefine per task class if needed.

## Sources

- Paper: *TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning* — Xu, Zhou, Sang, Li, Zhang, Du, Wang, Geramifard, 2026 — LinkedIn / Harvard / JHU.
- Related: process-reward-model line — see [prm](prm.md).
