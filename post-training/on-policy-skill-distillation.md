# On-Policy Skill Distillation (OPID)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A self-distillation auxiliary loss for agentic RL that turns sparse trajectory-level outcome rewards into *dense token-level* supervision — by mining "skills" directly from the agent's own completed on-policy trajectories. Removes the need for external skill memories or retrieved privileged context that drift out of distribution under the current policy.

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md), [_post-training.md](_post-training.md)
**Related:** [rejection-sampling.md](rejection-sampling.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [reasoning/prm.md](reasoning/prm.md)

---

## What it is

Outcome-only RL on long-horizon agents has a known pathology: a single scalar reward per trajectory provides no guidance on which intermediate decisions were good. The fashionable fix has been *skill-conditioned* self-distillation — give the policy access to a curated memory of "skills" (templates, sub-procedures, retrieved snippets) and distill against those during RL.

The problem: external skill memories and retrieved privileged context are expensive to maintain and are *off-policy* relative to whatever distribution the current policy actually induces. OPID's move is to mine the skill targets from the policy's *own* successful trajectories instead.

## How it works

- Run the policy on the RL task; collect completed trajectories with outcome rewards (standard rollout).
- For each successful trajectory, extract per-step "skill" supervision directly from the trajectory: e.g. the action distribution conditioned on the local sub-goal context.
- Build a per-token cross-entropy distillation loss against these self-mined targets.
- Add this loss as an auxiliary to the standard RL objective (GRPO / PPO with KL regularization). Every token now has a supervised signal in addition to the trajectory-level RL advantage.
- Targets are refreshed each round — they stay on-policy by construction.

## Why it matters

- **Removes the skill-memory infrastructure.** No external store, no retrieval pipeline, no privileged-context plumbing.
- **Dense signal without a separate PRM.** Token-level supervision falls out of the policy's own successful rollouts.
- **Multi-turn stability.** Matches or beats skill-memory baselines on multi-turn agentic benchmarks; on-policy mining sidesteps the drift problem.

## Gotchas & tricks

- Quality depends on having *some* successful trajectories; for hard tasks where the policy almost never succeeds, the self-mined targets are too few to be useful — combine with a warm-start SFT or rejection-sampled bootstrap.
- The auxiliary distillation weight is a tradeoff: too high and the policy reproduces its current mistakes; too low and the dense signal vanishes.
- Borderline: when the policy makes a stylistic shift, old self-mined targets become slightly stale within a few RL steps; the paper refreshes targets per-iteration to prevent this.

## Sources

- Paper: *OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning* — Yang, Wu, Lu, Shen, Zhang, Feng, Zhang, Luo, Lian, Wen, Tao, Tsinghua / ZJU / CUHK / NTU / Tongji, 2026 — arXiv:2606.26790.
