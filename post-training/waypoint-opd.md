# Waypoint-OPD (Waypoint-guided On-Policy Distillation)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation that reuses a teacher's **RL-induced policy shift** as a per-prefix learning signal, but *gates* that shift by whether the teacher's early and late training agree on it. A single endpoint delta can be the sum of early and late shifts that reinforce or partly cancel; comparing an intermediate teacher checkpoint to the start and end exposes the disagreement and suppresses the signal where directions conflict.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

Policy-shift distillation starts from an **unmodified base** teacher $\pi^{(0)}$ and the **RL-trained** teacher $\pi^{(T)}$, and uses their log-probability difference $\log\pi^{(T)} - \log\pi^{(0)}$ on student rollouts as a supervision signal. Transfers RL gains without replaying the teacher's RL run or storing its trajectories.

The failure mode this paper names: the *endpoint* shift collapses information. If the teacher's preference at prefix $x$ moved $+\Delta$ early then $-\Delta/2$ late, the endpoint shift ($+\Delta/2$) looks like a confident training signal, when in fact the teacher's own trajectory contradicted it. Waypoint-OPD inserts an intermediate checkpoint $\pi^{(w)}$ and uses the agreement between early ($\pi^{(0)}{\to}\pi^{(w)}$) and late ($\pi^{(w)}{\to}\pi^{(T)}$) shifts to gate the endpoint signal.

## How it works

For each student rollout prefix $(q, o_{<t})$, score the next-token distributions of three frozen teachers:

$$
s_t^{(0)} = \log\pi^{(0)}(\cdot\mid q,o_{<t}),\; s_t^{(w)},\; s_t^{(T)}
$$

Compute the two policy-shift vectors in log-prob space (relative, i.e. mean-centered over the vocab):

$$
\Delta_{\text{early}} = s_t^{(w)} - s_t^{(0)},\qquad \Delta_{\text{late}} = s_t^{(T)} - s_t^{(w)}
$$

Define the gate as their **cosine agreement** (or sign agreement on dominant actions):

$$
g_t = \max\!\left(0,\ \tfrac{\Delta_{\text{early}}\cdot\Delta_{\text{late}}}{\|\Delta_{\text{early}}\|\,\|\Delta_{\text{late}}\|}\right)
$$

The gated endpoint target is $g_t \cdot (s_t^{(T)} - s_t^{(0)})$, used as a per-prefix KL/distillation loss on the student. Cost over endpoint-only transfer: one extra frozen-teacher scoring pass per update step.

## Why it matters

- **Cheap filter on a cheap method.** On-policy distillation is already an order of magnitude cheaper than replaying the teacher's RL; Waypoint-OPD adds one scoring pass to filter noisy shifts, no new training objective.
- **Attacks a real failure mode.** Direction cancellation inside the teacher's RL run is invisible to endpoint distillation; the paper shows this isn't academic — removing or shuffling the gate degrades math reasoning.
- **Composes.** The gating is model-free with respect to the student and can be dropped onto other teacher-relative distillation recipes (reward-aligned reweighting, multi-teacher relative shifts, trajectory dropout).

## Gotchas & tricks

- **Checkpoint cadence.** The waypoint $w$ needs to actually see the teacher mid-training, not a cold-start checkpoint. The paper uses a single mid-run checkpoint; schedules with multiple waypoints are an obvious extension but not evaluated.
- **Normalize before comparing.** Raw log-prob deltas inherit the LM-head bias; comparing them in *relative* space (centered per prefix) is what makes "direction agreement" meaningful.
- **One extra scoring pass, not three.** The base and endpoint teachers are already needed; only the waypoint teacher adds overhead.
- **Verified on reasoning, not open chat.** All six teacher-student settings are math + code with Qwen3 teachers. Preference-RM-trained teachers may have smoother shifts where the gate is less informative.
- **Ablations.** Removing the gate (always 1) hurts; shuffling gates across prefixes hurts more — so the information is in *which prefix* gets gated, not just the gate's average.

## Sources

- Paper: *Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation* — Hou, Jiang, Quan, Li, Feng, Liu, Zhang (Alibaba Cloud), 2026 — [arXiv:2610.21601](https://arxiv.org/abs/2610.21601).
- Code: https://github.com/hhhbxloveai/WaypointOPD
- Related concurrent work: *Reward-Aligned Reweighting for On-Policy Distillation* (2609.35517); *Composing What Each Teacher Learned* (2610.10460); *Residual Advantage: Student-Relative Teacher Guidance for RLVR* (2610.11519); *The Teacher Is a Direction, Not a Destination* (2609.36484).
