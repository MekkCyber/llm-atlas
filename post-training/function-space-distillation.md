# Function-Space Policy Optimization (FSPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A self-distillation variant for reasoning RL that stores a **sparse set of historical next-token distributions** from the student's own earlier checkpoints and uses them as **fixed teachers**. Instead of distilling against a moving self-teacher (which is unstable) or a full replay buffer (which is expensive), FSPO freezes a small set of function-space targets. On Olmo3-7B-Instruct it reports **+4 avg accuracy over the strongest baseline under a 5-hour budget** and reduces peak memory from **~70 GB → 56 GB**.

**Prereqs:** [rlvr.md](rlvr.md), [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [on-policy-self-training.md](on-policy-self-training.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [_rewards.md](_rewards.md)

---

## What it is

Reasoning RL with verifiable rewards (RLVR / GRPO) gives sparse, outcome-level supervision. Adding a token-level self-distillation term helps the model, but the teacher usually tracks the student, so the targets move with every update. FSPO fixes this by *storing snapshots of past token distributions* and distilling against those fixed points.

## How it works

- During training the student periodically **snapshots its own next-token distributions** on a small anchor set of inputs. Snapshots are sparse — not every step, not every token — enough to span the student's learning trajectory.
- At each training step, the policy objective is **RL (outcome reward)** + **a function-space self-distillation loss** that matches the student's current distributions to the stored snapshots on the anchor set.
- Because the snapshots are frozen, there is no moving-teacher instability and no need to keep multiple full teacher checkpoints in memory. The sparse snapshot structure keeps the memory overhead bounded.

The practical consequence: a token-level signal is added on top of the RL reward without the implementation cost of PPO's value network or the stability cost of a running EMA teacher.

## Why it matters

- **Stability without replay.** Resolves the main pathology of self-distillation (teacher drift) without the compute of ensembling multiple checkpoints or the memory of a full replay buffer.
- **Memory win is real.** The ~20% peak-memory reduction (70 → 56 GB on a 7B) makes RLVR training cheaper per step and lets batch sizes grow.
- **Composable with GRPO/RLVR.** FSPO drops in as an auxiliary loss; nothing about the group-relative outcome reward changes.

## Gotchas & tricks

- **Snapshot schedule is a hyperparameter.** Too few snapshots → the function-space signal is weak; too many → memory savings disappear. The paper's recipe uses a sparse geometric-ish schedule.
- **Anchor set must stay informative.** If the anchor inputs become trivial as the student improves, the distillation term goes to zero and only the RL signal remains. Rotate anchors as training progresses.
- **Not a replacement for RL reward.** FSPO is complementary. Used alone (no outcome reward) it just encourages the student to mimic its own past, which is a null gradient.

## Sources

- Paper: *Function-Space Temporal Self-Distillation for Policy Optimization* — Bhattacharya & Verma, IIT Kanpur, 2026 — https://arxiv.org/abs/2610.19687
