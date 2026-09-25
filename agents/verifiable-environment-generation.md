# Verifiable Environment Generation (VHD-Play)
*Depth — solve a mechanism first, then instantiate it as an agentic RL environment.*

**TL;DR:** Existing agentic-RL environment generators build the environment first and try to define the reward and reference trajectories afterward — so dynamics, evaluation, and rollouts have to be aligned post hoc. **VHD-Play reverses the pipeline:** sample and *solve* a mathematical model, then render its decision process as stateful tools. Executable dynamics and the trajectory-scoring reference both come from the same solved model, so verifiable rewards are correct by construction. Produces 3,300 diverse environments at a few cents each. Training Qwen3.6-35B-A3B on three environment families raises its mean agentic score from **0.204 → 0.815**, with gains generalizing to eight unseen mechanism families and external benchmarks.

**Prereqs:** [rlvr](../post-training/rlvr.md), [grpo](../post-training/grpo.md)
**Related:** [jit-memory](jit-memory.md)

---

## What it is

An environment-generation pipeline for training LLM agents with RL. The pipeline treats each new environment as a **solved instance of a mathematical mechanism** — a game, an optimization problem, a control problem — with a known analytic (or algorithmic) solution. The environment shell wraps the mechanism as a set of stateful tools the agent can call; the reference scoring function comes directly from the solved model, not from a separate annotator.

## How it works

Pipeline per environment:

1. **Sample a mechanism.** A "setter" LLM (35B in the paper) samples a mathematical model from a mechanism family — e.g., a scheduling problem, a bandit variant, a routing problem — parameterized by size, difficulty, and structure knobs.
2. **Solve it.** The setter either derives a closed-form solution, calls a solver, or produces a program that computes the optimum. The solution defines the analytic reward and the reference trajectory.
3. **Render as stateful tools.** The mechanism is exposed to the agent through a small set of stateful API endpoints (query state, take action, observe result). Written- and stateful-parameter versions coexist.
4. **Score with the solved reference.** Rollout scoring uses the solved model — either the analytic optimum or the reference trajectory — so reward is verifiable and cheap. This is RLVR at the environment level.

Training loop: sample a batch of these environments, run GRPO rollouts on each, treat the solved-model score as the terminal reward.

## Why it matters

Environment supply is a bottleneck for agentic RL: hand-authored environments are expensive and homogeneous. VHD-Play makes environment generation **near-free** (cents per environment) while guaranteeing that the reward and reference are consistent with the environment's dynamics — no post-hoc annotation, no RM training. The comparison between written-out problems and stateful versions in the paper shows that **most of the learnable gap is in stateful interaction**, not in problem solving per se, so training on stateful mechanisms is where agent capability actually improves.

## Gotchas & tricks

- The setter model has to solve the sampled mechanism correctly. A 35B setter is enough for the mechanism families used; scaling to harder mechanisms will need larger setters or symbolic solvers on the side.
- Difficulty control is a real dial — sampled mechanisms have to bracket the current agent's skill level for RL gradient to be nonzero (Goldilocks curriculum). The paper reports scale-matched training retains gains as mechanism size grows.
- Generalization to external benchmarks (E-Commerce Bench, travel planning, function calling) suggests the stateful-tool exposure is what transfers, more than any specific mechanism.

## Sources

- Paper: *Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms* — Shen et al., 2026 — [arXiv:2609.27321](https://arxiv.org/abs/2609.27321)
