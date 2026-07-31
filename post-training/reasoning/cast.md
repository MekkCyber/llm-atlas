# CAST — Credit Assignment from Solver Teachers

*Depth — turn-level RL advantages from a domain-solver's value function, equivalent to on-policy distillation from the solver.*

**TL;DR:** RLVR on long-horizon tasks suffers from sparse terminal rewards — nothing tells the agent which decision mattered. CAST plugs in a **game solver** (Sokoban, Minesweeper, Rush Hour) as a value oracle: the *change in solver value* between consecutive states becomes a turn-level advantage the RLVR loop consumes as extra credit. Under a soft-optimal-solver assumption, maximizing that solver advantage is provably equivalent to on-policy distillation from the solver — but requires only scalar state values, not teacher logits.

**Prereqs:** [../rlvr.md](../rlvr.md), [../grpo.md](../grpo.md), [prm.md](./prm.md), [orm.md](./orm.md)
**Related:** [../../agents/README.md](../../agents/README.md), [long-cot-rl.md](./long-cot-rl.md)

---

## What it is

A dense turn-level credit signal for [RLVR](../rlvr.md), sourced from a **solver** the researcher already has for the environment (search-based, exact, or learned). Instead of learning a [PRM](./prm.md) from scratch or accepting the sparse terminal reward of [ORM](./orm.md), CAST reads off the solver's state value $V^*(s_t)$ and turns the step-by-step delta into a turn-level advantage.

## How it works

At each turn $t$ in an agent trajectory:

1. **Solver value at each state:** $V^*(s_t)$ from a domain-specific solver (BFS/A*, symbolic solver, or trained value network).
2. **Turn-level solver advantage:** $A^{\text{CAST}}_t = V^*(s_{t+1}) - V^*(s_t)$ — positive if the action advanced the state toward the goal.
3. **Inject into the RLVR loss:** add $A^{\text{CAST}}_t$ to the standard [GRPO](../grpo.md) / PPO advantage, either as a shaped reward at step $t$ or as a direct turn-level term.

**Equivalence to on-policy distillation.** Under a soft-optimal solver assumption (policy proportional to $\exp(V^*/\tau)$), maximizing the solver advantage is equivalent to minimizing the KL between the student policy and the solver policy on states the student actually visits. That's on-policy distillation — but implemented with **scalar values**, not the teacher's action distribution. This is what makes CAST cheap: no full teacher logits per step.

## Why it matters

- **Dense turn-level credit without a learned PRM.** [PRM](./prm.md) training needs step-annotated data; CAST reads a signal off an existing solver. Cheaper wherever a solver exists.
- **Transfer beyond the trained games.** In the paper, an agent trained on three puzzle games (Sokoban, Minesweeper, Rush Hour) achieves the highest zero-shot average on **ALFWorld** and **WebShop** — evidence the recipe teaches an interaction-shape prior, not just game-specific tricks.
- **Template for verifier-rich domains.** Any domain where a solver / simulator / verifier can score partial progress fits the same shape — coding-with-tests, math-with-Lean, tool-use with executable ground truth.

Reported result: SoTA on all three training games under in-domain and unseen-difficulty splits; best zero-shot average on ALFWorld and WebShop.

## Gotchas & tricks

- **The soft-optimal assumption is doing real work.** If the solver's induced distribution is far from soft-max-over-values (e.g. an exact solver picks one action deterministically), the equivalence to on-policy distillation is only heuristic. Empirically fine on the paper's games; treat as a working assumption elsewhere.
- **Needs a solver.** Domains without one (open-ended dialog, creative writing) can't use CAST directly. Alternatives: [PRM](./prm.md) training or [ORM](./orm.md) with denser terminal rewards.
- **Solver quality caps agent quality.** If $V^*$ is noisy or wrong on some states, the agent inherits the noise. Use CAST with solvers you'd bet on.
- **Combine, don't replace.** Best results come from adding the CAST advantage to the standard [GRPO](../grpo.md) terminal-reward advantage, not replacing it. Turn-level signal densifies; terminal reward keeps the objective grounded.
- **Scalar-only vs full-logit distillation.** If you *do* have teacher logits, full on-policy distillation is stronger. CAST is the scalar-only relaxation for when logits aren't accessible (or the teacher isn't an LLM at all).

## Sources

- Paper: *CAST: Game Solvers as Turn-Level Teachers for LLM Agents* — Wang et al., USTC / Nanjing U. / Wuhan U. / Meituan, 2026 — introduces the method and its distillation equivalence. See [../../daily-papers/2026-07-30.md](../../daily-papers/2026-07-30.md).
- Related: [PRM](./prm.md), [ORM](./orm.md), [RLVR](../rlvr.md), [GRPO](../grpo.md).
