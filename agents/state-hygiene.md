# State Hygiene for LLM Agents (AEWM Pattern)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A world-modeling approach for LLM agents that does **not** try to predict future observations (they'll be observed anyway), and instead actively curates the agent's *belief state*: an **Action Judge** filters unsupported actions before they run, and **State Revision** rewrites the trajectory history to drop assumptions later observations contradicted. Reduces the looping / stale-context failures typical of ReAct-style long-horizon agents.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Classical world models predict the next observation given a state and action; for LLM agents equipped with real tools, that's low-value work — the environment will produce the observation anyway, and the high-entropy content of tool responses is hard to predict. The failure mode LLM agents actually suffer from is **task-state contamination**: unsupported assumptions the agent stated three turns ago persist in the context and distort every subsequent decision.

The AEWM pattern replaces observation prediction with two lightweight mechanisms operating on a maintained belief summary:

- **Action Judge** — before the agent commits an action, a critic head (same model, different prompt) scores whether the action is supported by the current belief state. Unsupported or contradicted actions are rejected without ever hitting the environment.
- **State Revision** — after each observation, another pass rewrites the belief summary to remove statements the new observation invalidates. The trajectory history is then re-emitted with the corrected summary in place of the corrupted one.

## How it works

Per agent turn:

1. Belief state $b_{t-1}$ = a natural-language summary of what the agent believes about the task.
2. Agent proposes action $a_t$ given $b_{t-1}$ and the user goal.
3. **Action Judge**: score $J(a_t \mid b_{t-1}) \in \{\text{support}, \text{unsupported}, \text{contradicted}\}$. If not "support", regenerate $a_t$ (up to a retry cap).
4. Execute $a_t$, observe $o_t$.
5. **State Revision**: compute $b_t = \text{revise}(b_{t-1}, a_t, o_t)$ — an edit, not a rewrite. Contradicted claims are struck through; new observations added; unchanged claims kept.
6. Continue.

The system is trained end-to-end (SFT + light RL) on trajectories with human-annotated revision targets.

## Why it matters

- **A concrete alternative to observation prediction.** For agents with real tool access, belief maintenance is where the modeling budget should go, not next-observation.
- **Fights the drift failure mode.** ReAct-style agents fail on long horizons because assumptions from turn 3 poison decisions on turn 30; State Revision directly targets that.
- **Composable.** Action Judge and State Revision are prompt-level primitives — they slot into any existing agent scaffold without retraining the base model.

## Gotchas & tricks

- **Belief-state format is load-bearing.** A free-form summary is hard to revise reliably; the paper uses a structured schema (facts, hypotheses, open questions) which makes contradiction-detection tractable.
- **Judge budget.** Naive Action Judge doubles latency per action. Batch multiple candidate actions and judge them jointly, or gate on confidence.
- **Overzealous revision loses context.** If Revision aggressively drops claims, the agent forgets grounded facts and re-does work. The paper's default is "strike through, don't delete" for auditability.
- **Not a substitute for tool selection.** State hygiene helps *reason* about tool outputs; it doesn't reduce the number of tools the agent must know about.

## Sources

- Paper: *Agent-Editing World Model: Rethinking World Modeling for LLM Agents* — Sun, Chen, Meng, et al., Renmin University of China, 2026, [arXiv 2609.28416](https://arxiv.org/abs/2609.28416) — introduces the AEWM pattern with Action Judge and State Revision.
