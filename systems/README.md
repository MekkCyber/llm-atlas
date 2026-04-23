# Systems

*The training-time backbone — orchestration, distributed scheduling, fault tolerance, and the infra patterns that keep thousand-GPU jobs running for weeks.*

---

## What This Is

This folder is about the *training-time* infrastructure: how compute is scheduled across many machines, how state survives failures, how rollouts are generated and aggregated for RL, and how all of it is glued together. Distinct from [inference/](../inference/), which is about serving-time concerns.

---

## What Belongs Here

- **Distributed primitives** — collectives (allreduce, allgather), NCCL, communication patterns.
- **Orchestration** — Ray, Kubernetes, custom schedulers, actor models.
- **Fault tolerance** — node failure, elastic training, checkpoint-restart.
- **Checkpointing** — sharded state, async checkpoints, recovery time.
- **Rollout systems** — generating samples for RL post-training at scale.
- **Job lifecycle** — submission, monitoring, debugging multi-node jobs.
- **Hardware topology** — racks, NVLink, IB, awareness in scheduling.

## Reading Order

1. Distributed primitives & collectives
2. Orchestration with Ray
3. Checkpointing strategies
4. Fault tolerance & elastic training
5. Rollout systems for RL

---

## Concept Pages (depth)

- [Ray](ray.md)
- [DualPipe](dualpipe.md)
- [Partial Rollouts](partial-rollouts.md) — Kimi k1.5's long-context RL rollout infrastructure (async rollout workers, replay-buffer continuation, loss masking for stale segments).

---

## Related

- [pre-training/](../pre-training/) — parallelism strategies that use this infra.
- [post-training/](../post-training/) — RL rollouts run on top of this.
- [inference/](../inference/) — serving-time infra (different concerns, shared primitives).
