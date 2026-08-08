# AgentOPSD
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A critic-free credit-assignment scheme for multi-turn agentic RL. Aggregates token-level teacher–student log-probability gaps into per-turn evidence, then does a recursive Bayesian update in log-odds space to identify which turns actually drove the outcome. Drops into GRPO without extra rollouts or a value head.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [rejection-sampling.md](./rejection-sampling.md) (self-distillation intuition).
**Related:** [rlvr.md](./rlvr.md) · [rl-prompt-curation.md](./rl-prompt-curation.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [../agents/README.md](../agents/README.md)

---

## What it is

Long-horizon agent RL sees only a sparse outcome reward at the end of a trajectory, but the trajectory may span 10–50 turns of tool calls, observations, and reasoning. GRPO broadcasts the same normalized advantage to every token — a coarse signal that credits pivotal turns and filler turns equally. AgentOPSD replaces this uniform assignment with a **turn-level advantage weight** derived from a teacher.

## How it works

Let $\pi_S$ be the student policy, $\pi_T$ a "privileged" teacher (e.g. the student with the correct answer prefilled, or a stronger model that saw the outcome).

**Step 1 — token → turn evidence.** For each turn $k$ compute the mean log-prob gap:

$$
e_k = \tfrac{1}{|o_k|}\sum_{t\in o_k}\bigl(\log \pi_T(o_{k,t}\mid \cdot)-\log \pi_S(o_{k,t}\mid \cdot)\bigr)
$$

Large $e_k$ = the teacher would have said something noticeably different here = this turn matters.

**Step 2 — recursive Bayesian belief in log-odds space.** Maintain a running belief $L_k$ that "turn $k$ is pivotal" as a log-odds ratio, updated by $L_k = L_{k-1} + \alpha\cdot e_k$. History-dependent: the same $e_k$ contributes more when preceded by uncertain turns.

**Step 3 — turn weights.** Convert $L_k$ into a soft weight $w_k = \sigma(L_k)$ and reweight the GRPO per-token advantages inside turn $k$ by $w_k$. Pivotal turns get a stronger push; filler turns get a weaker push. Objective and clipping are otherwise standard GRPO.

No new rollouts, no value network — only forward passes of the teacher on the collected trajectories.

## Why it matters

- **Cleaner credit than uniform-broadcast GRPO** in multi-turn agent settings where a few decisions decide the outcome.
- **Critic-free.** Sidesteps value-network warmup pathologies that plague PPO in agent RL.
- **Composable.** Slot into any GRPO-derived pipeline (ARPO, T-GRPO, etc.) as a per-turn reweighter.
- Reaches 89.1% success on ALFWorld with Qwen2.5-7B, beating GRPO and strong self-distillation baselines.

## Gotchas & tricks

- **Teacher choice matters.** Prefilled-answer teacher is cheap and privileged; a stronger open model is more permissive but noisier. Paper uses the former.
- **$\alpha$ controls history sensitivity.** Too high and one loud turn dominates the trajectory; too low and the update degenerates to a per-turn independent reweighter.
- **Softmax collapse.** If $w_k$ concentrates on one turn, gradient signal from the rest vanishes. Normalize $w_k$ across the trajectory or clip.
- **Only meaningful when trajectories have real turn structure.** For single-turn tasks (reasoning, code) this collapses to per-response advantage and buys nothing.
- **The teacher log-probs are computed offline** on the already-collected rollouts — cost scales with dataset size, not with rollout width.

## Sources

- Paper: *AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning* — Wang et al., Tsinghua/ZJU/Meituan, 2026 — [arXiv:2608.05987](https://arxiv.org/abs/2608.05987).
- Related: DeepSeekMath (GRPO), 2024 — the base algorithm this reweights.
