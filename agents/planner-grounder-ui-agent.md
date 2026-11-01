# Planner–Grounder UI Agent (GiGPPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Apple's recipe for training a UI-navigation agent with online RL on live iOS apps. The agent is split into a *planner* (textual plan output, trained with PPO) and a *grounder* (plan → pixel coordinates, SFT-trained and frozen during RL). An LLM judge supplies a dense per-step progress reward plus a terminal task-completion reward. **GiGPPO** augments PPO's value estimate by borrowing outcomes from *visually similar screens* across parallel rollouts. On UINavBench, 8B agents match Gemini-3.1-Pro; transfer to Android gives ~7× the SFT baseline's success rate.

**Prereqs:** [README.md](README.md), [../post-training/ppo.md](../post-training/ppo.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/_rewards.md](../post-training/_rewards.md)

---

## What it is

Prior UI-agent training was SFT (memorizes trajectories, fragile under distribution shift) or terminal-only RL (reward too sparse to converge at reasonable budget). This paper combines three ideas: split the agent, use an LLM judge as a dense reward, and share value estimates across visually similar screens.

## How it works

**Split architecture.**
- **Planner $\pi_\theta$:** input = current screen + task instruction + plan history. Output = a textual plan for the next step. Trained with PPO.
- **Grounder $g_\phi$:** input = plan + current screen. Output = pixel coordinates for the action. SFT-trained from recorded human trajectories; **frozen during RL**.

RL gradient only flows through the planner. The grounder is a learned renderer of plans to actions.

**Dense reward from an LLM judge.** At each step, an LLM reads (plan, before-screen, after-screen) and emits (a) a scalar progress reward and (b) a terminal reward when the task is complete. Reward is sparse-becomes-dense: terminal check + per-step credit.

**GiGPPO value sharing.** Standard PPO estimates value $V(s)$ from the current trajectory's rewards. GiGPPO identifies screens visually similar to $s$ in *parallel* rollouts and borrows their observed returns to improve the value estimate for $s$ — a cross-trajectory variance-reduction trick that works because UI apps have many near-duplicate screens (settings menus, confirmation dialogs). The value target becomes a weighted combination of own-trajectory returns and similar-screen returns.

**Infrastructure.** Hundreds of concurrent virtual iOS devices feed rollouts. Trained 8B planners.

## Why it matters

Three concrete lessons: (a) separating planning from grounding lets you train the hard part (planning) with sparse-but-rich signal and the easy part (grounding) with supervised data; (b) LLM-judge dense rewards tame the terminal-only sparsity problem on realistic apps; (c) in domains with structural screen repetition, cross-trajectory value sharing is a free variance reduction. The 8B planner matching Gemini-3.1-Pro on UINavBench, with ~7× Android transfer, is strong evidence the recipe generalizes beyond iOS.

## Gotchas & tricks

- **Grounder stays frozen during RL.** If you let the grounder train jointly, planner gradient paths through pixel coordinates destabilize PPO.
- **Judge calibration matters.** An inconsistent LLM judge produces non-stationary rewards; the paper uses deterministic decoding and few-shot calibration.
- **Visual similarity for value sharing is cheap** — perceptual-hash or CLIP-embedding nearest-neighbor on parallel rollout screens.
- **Scales with parallel rollout count.** More parallel iOS devices → more visually-similar neighbors → tighter value estimates. The paper's "hundreds of devices" setting is the empirical sweet spot.
- **Transfer to Android is dialect-sensitive.** Grounders trained on iOS screens degrade on Android layouts; the paper's ~7× Android gain comes from an Android-fine-tuned grounder but the same iOS-trained planner.
- **Not a world-model agent.** No next-screen predictor; purely reactive planning conditioned on history.

## Sources

- Paper: *Online Dense Reinforcement Learning with Planning-Grounding Agent for UI Navigation on Live iOS Apps* — Duan, Gatmiry, Nichols, Bengio, Abbe — Apple, 2026 — [arXiv:2610.39264](https://arxiv.org/abs/2610.39264).
- Reference: *UINavBench* — the UI-navigation benchmark used for comparisons.
