# Reflective Rulebook World Models (Memento 3)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Frozen LLM agents learn an explicit, inspectable world model by maintaining a natural-language **rulebook** of hypotheses that is also compiled to **executable code**. Prediction errors drive refinements, accepted only when (a) the LLM certifies the code faithful to the rulebook and (b) replay reproduces observed transitions. A population extension carries several world models in parallel. Clears all 25 public ARC-AGI-3 games with mean RHAE 100.0 and 44% of the human action count.

**Prereqs:** [README.md](./README.md), [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)
**Related:** [language-world-models.md](./language-world-models.md) · [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)

---

## What it is

A write-run-verify loop for recursive self-improvement on top of a *frozen* base LLM. The agent's "world model" is stored twice: once as a human-readable rulebook, once as compiled prediction/planning code. The dual representation keeps the model debuggable while making it usable by symbolic planners.

## How it works

- **Rulebook.** Natural-language entries of the form "when X in state S, Y happens." Revisable at every step.
- **Compilation.** The LLM translates the rulebook into executable code — a transition function the planner can call.
- **Refinement loop.** When prediction disagrees with observation:
  1. The LLM proposes a rulebook edit.
  2. The LLM certifies the compiled code is faithful to the edited rulebook (LLM-as-judge).
  3. Replay the recent trace against the new code; accept the edit only if observed transitions reproduce.
- **Population extension.** Maintain several candidate world models in parallel; selection is based on recent prediction accuracy, letting the agent hedge during exploratory phases.

Planning uses the compiled model (any search / MPC-style procedure); no further LLM calls at execution time once the model is good enough.

## Why it matters

A concrete recipe for recursive self-improvement that does **not** require fine-tuning: the base LLM stays frozen, the agent gets better by editing an external, inspectable model of the world. The ARC-AGI-3 clear (all 25 public games, mean RHAE 100.0 at 44% of human actions) and an Atari-Pong case where the trained compiled controller wins 21:0 across three episodes with no further LLM calls both argue the loop produces durable artefacts, not just prompt-level improvements.

## Gotchas & tricks

- The dual representation only works if the LLM's rulebook↔code compilation is reliable. Weak LLMs break the certification step.
- Rulebooks grow; prune or they drown the compiler. The paper's acceptance test provides implicit pressure.
- Population size trades robustness for compute. One model is faster to converge; several hedge against local optima.
- Not a general planner — the method shines where a compact rule set exists (puzzle games, Atari). Open worlds with no learnable rulebook stay hard.

## Sources

- Paper: *Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks* — Zhao et al., UCL / Huawei Noah's Ark Lab / U. Liverpool, 2026 — [arXiv:2610.11794](https://arxiv.org/abs/2610.11794)
