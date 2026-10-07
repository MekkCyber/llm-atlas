# Online Agentic Test-Time Training (ASCENT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A deployed agent produces trajectories; at the end of each task, a verifier emits one signal. **ASCENT** (Lu & Gong, 2026) turns those verified trajectories into **persistent LoRA weight updates** via self-distillation: the model's own frozen initial copy acts as a hindsight-privileged teacher that re-predicts next-token distributions along the verified trajectory, and the live policy distills them. No stronger teacher, no replay buffer, no external reward model.

**Prereqs:** [../post-training/test-time-training.md](../post-training/test-time-training.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md) · [../post-training/grpo.md](../post-training/grpo.md) · [../evaluation/osworld-pro.md](../evaluation/osworld-pro.md)

---

## What it is

In-context agent-adaptation stacks (memories, reflections, skill libraries) adapt via retrieval over a frozen policy. ASCENT instead **updates the policy's weights** from the stream, in a single pass, during deployment. One pass means: each task is executed once; its verified trajectory is the only learning signal; the update must survive before the next task begins.

Directly imitating or reinforcing single-attempt rollouts is unstable (one example is a weak gradient, and self-feedback drift is real — see [../post-training/test-time-training.md](../post-training/test-time-training.md)). ASCENT's answer is **hindsight self-distillation**: use the trajectory's *verification outcome* as privileged information fed to a frozen copy of the model, then distill that copy's token-level distributions into the live policy.

## How it works

- **Rollout.** The live policy $\pi_\theta$ executes the task; verifier returns success/fail at termination.
- **Hindsight privileged teacher.** The **frozen initial copy** $\pi_0$ receives the verified trajectory with its outcome as privileged context (not available at the time of rollout) and predicts next-token distributions $\pi_0(\cdot \mid \text{trajectory}, \text{outcome})$ along it. The hindsight framing lets $\pi_0$ act as a stronger teacher than the live policy without actually being a different model.
- **Self-distill into LoRA.** Train LoRA fast-weights on $\pi_\theta$ to match those distributions on the verified trajectory. Updates persist across tasks.
- **Trajectory cleanup.** Invalid-action turns (steps the environment rejected) are stripped from the trajectory before distillation.

## Why it matters

- Lets a deployed agent **keep learning from its own successes**, without a replay buffer, reward model, external teacher, or server-side retraining.
- Beats memory-retrieval online-adaptation on **ALFWorld, WebShop, AppWorld** across model scales and transfers to **held-out scenes**.
- Uses LoRA as the persistence mechanism — cheap, isolatable, revertible — which makes the approach safer to deploy on top of a shared base model.

## Gotchas & tricks

- **Verifier matters.** Garbage-in-garbage-out: ASCENT's gains come from the verification signal. Verifier-less tasks need an alternative outcome gate (e.g., LLM-judge).
- **Hindsight teacher ≠ an oracle.** $\pi_0$ with privileged outcome is still just the initial model; on tasks the initial model fundamentally cannot solve, the teacher distribution isn't informative, and distillation plateaus.
- **Guard against self-feedback.** Although verification plus privileged outcome dampens drift, long streams still risk the TTT self-feedback failure — periodically probing on held-out tasks (Settlement-style) is a cheap safety net.
- **LoRA rank is a knob.** Too low → under-adapts; too high → risk of forgetting the base. Paper uses modest ranks.

## Sources

- Paper: *ASCENT: Online Test-Time Training of Long-Horizon Agents via Self-Distillation of Verified Experience* — Haodong Lu, Dong Gong, 2026 — [arXiv:2610.05303](https://arxiv.org/abs/2610.05303).
- Related: TTT self-feedback failure — [../post-training/test-time-training.md](../post-training/test-time-training.md).
