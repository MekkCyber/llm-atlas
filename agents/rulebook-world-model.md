# Rulebook World Model
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A model-based self-improvement loop for LLM agents that keeps the *world model* outside the network: a natural-language **rulebook** of hypotheses about an environment is compiled into executable code, the code's predictions are verified against new observations, and discrepancies drive rule revision. The policy LLM stays frozen; learning lives in the rulebook. Introduced as *Memento 3* (2026); on ARC-AGI-3 one agent clears all 25 public games, and in an Atari Pong case study a compiled controller wins 21:0 across three evaluated episodes with no LLM calls at inference.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [code-world-simulation.md](code-world-simulation.md)
**Related:** [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md) · [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

A model-based reinforcement-learning approach where the world model is a **natural-language rulebook** rather than a neural network. The rulebook captures hypotheses about environment dynamics ("pressing A moves the paddle up by `k`"); a code-compilation step turns those rules into an executable simulator; the simulator's predictions are compared against the agent's observations; mismatches trigger reflection and rule revision.

## How it works

- **Observe.** The frozen LLM agent interacts with the real environment and keeps a trace.
- **Reflect.** The agent examines the trace in light of the current rulebook, flags predictions that diverged from reality, and proposes rule edits.
- **Revise.** Edits (insert, modify, delete, generalize, specialize) are applied to the rulebook. The rulebook stays small and human-readable.
- **Compile.** The LLM compiles the rulebook into executable code — a deterministic simulator of the environment.
- **Verify.** New observations are checked against the simulator's predictions; the loop closes.
- **Population extension.** Multiple candidate rulebooks are maintained in parallel; weak ones are pruned.

Once the rulebook is accurate, the compiled controller can be executed without any further LLM calls — behavior is distilled into code.

## Why it matters

- Learning is *inspectable*: a human can read the rulebook, add or remove rules, or reject a proposed revision.
- Decouples capacity to reason (frozen LLM) from capacity to specialize (external memory); supports long-horizon tasks without updating weights.
- Allows "no-LLM" deployment: once compiled, the controller is cheap, deterministic, and auditable.

## Gotchas & tricks

- Rule-compilation errors are the main failure mode; the paper's loop sidesteps them by executing the compiled code and treating runtime exceptions as refutations.
- The LLM's bias toward confident wrong rules is a problem — population-level diversity (multiple rulebooks) is a load-bearing defense.
- Rulebook growth is unbounded in theory; the authors keep it legible by periodically compacting via LLM-driven consolidation.
- Works well where the dynamics are *symbolic enough to be writeable in a few hundred lines*. Rich visual environments still need upstream perception.

## Sources

- Paper: *Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks* — Zhao, Yu, He et al., UCL / Huawei Noah's Ark Lab (UK) / Liverpool, 2026 — [arXiv:2610.11794](https://arxiv.org/abs/2610.11794).
