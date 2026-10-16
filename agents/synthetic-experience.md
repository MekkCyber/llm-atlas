# Synthetic Experience (for Agent Post-Training)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use **simulated games or sandbox environments** as a source of large-scale, auto-checkable action trajectories for post-training computer-use / tool-use agents. The game is the verifier — rewards are extracted programmatically from game state, so GRPO-style RL has clean signals without hand-labeled data.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [README.md](./README.md) · [../post-training/_distillation.md](../post-training/_distillation.md)

---

## What it is

Agent RL is bottlenecked by **reward design**. For real tasks (web navigation, code editing, chat), writing automatic correctness checkers is hard. The standard workaround is a learned RM or an LLM judge, which brings its own hacking problems.

Synthetic experience sidesteps this: pick domains where correctness is **mechanical** — games, puzzles, simulated environments with explicit state and goal predicates — and let the environment itself produce the reward signal. For agent post-training, this reframes "need better reward" as "pick a different data source."

## How it works

The template (Player-1 is the concrete instance):

1. **Environment set.** Curate $N$ synthetic environments spanning action spaces similar to the deployment target (e.g., keyboard/mouse control, text commands, GUI widgets). The paper uses 16 2D and 3D environments.
2. **Reward instrumentation.** Each environment emits a programmatic reward signal — task-complete predicate, score delta, geometric progress — automatically from engine state.
3. **SFT warm-start.** Gather curated gameplay recordings (humans or scripted agents) and SFT the computer-use base model on them. Fixes the "the model doesn't know what the screen is" cold-start problem.
4. **RL stage.** Run GRPO (or similar) with the environment reward. $G$-way sampling per state; standard GRPO loss.
5. **Transfer evaluation.** Test on real tasks (Minecraft tasks, out-of-distribution games) to measure transfer.

## Why it matters

- **Scalable reward engineering.** Simulators give you unlimited auto-checkable trajectories. Compared to hand-authored RM datasets, the ceiling is much higher.
- **Transfer to real tasks.** The reported Minecraft task improvements and OOD-game transfer suggest the learned priors generalize — the method isn't just gaming a specific environment.
- **Composable with distillation.** Synthetic rollouts can be filtered via [rejection sampling](../post-training/rejection-sampling.md) and used for SFT in heterogeneous student stacks.
- **The data lesson.** Where to spend post-training compute on an agent: on an environment set the agent will not deploy in, if those environments produce the right kind of correctness signal. "Train on games, deploy on tools."

## Gotchas & tricks

- **Environment-task gap.** Transfer only works if the synthetic action space is close enough to the real one. Keyboard-mouse games generalize to desktop apps; chess doesn't generalize to anything useful.
- **Reward hacking in the environment.** Any programmatic reward is hackable (exploit a bug, pathological state); shape rewards with sanity constraints.
- **Diversity matters more than scale.** 16 environments beats 1 environment with 16× the rollouts — the paper's design choice.
- **SFT warm-start is non-optional.** Pure RL from the base model in a game environment is a sparse-reward disaster. Short SFT on recordings gets the model into the reward basin.
- **Evaluation discipline.** The whole point is transfer. Measure on tasks the model never saw at training time; in-domain training-env scores can look great and mean nothing.

## Sources

- Paper: *Player-1: Learning to Play from Synthetic Experience* — Ouyang, Wang, Hu, Yin, Sun, Wang, Pu, Cheng, Chen, Shan, Shou, Ng (NUS / Tencent IEG / NTU), 2026 — [arXiv:2610.20584](https://arxiv.org/abs/2610.20584). 16 synthetic 2D/3D environments; Minecraft task gains and OOD-game transfer.
