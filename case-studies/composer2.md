# Case Study: Composer 2

*Anthropic's RL post-training system for Claude — a distributed pipeline that orchestrates policy optimization, rollout generation, tool execution, and sandboxed environments at scale.*

**Related concepts:** [Post-training](../post-training/) · [Ray](../systems/ray.md) · [Environments](../agents/environments.md) · [Fault tolerance](../systems/fault-tolerance.md) · [Checkpointing](../systems/checkpointing.md)

---

## Overview

Composer 2 is the system Anthropic uses to perform RL post-training on Claude models. It takes a pretrained (or instruction-tuned) model and refines it through iterative policy optimization — generating responses, scoring them, and updating the policy to improve along multiple reward signals.

What makes Composer 2 interesting is not the RL algorithm itself (variants of PPO / RLHF are well-understood) but the **systems architecture** required to make it work at scale:

- Thousands of rollouts per training step, each potentially involving multi-turn tool use
- Sandboxed execution environments for code, bash, and other tools
- Distributed orchestration across hundreds of workers
- Fault tolerance for long-running training jobs (days to weeks)
- Heterogeneous workloads: GPU-heavy policy updates mixed with CPU-heavy rollout generation

The core insight: **RL post-training at scale is fundamentally a distributed systems problem**, not just an ML problem.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        COMPOSER 2                                 │
│                                                                   │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐  │
│  │             │     │              │     │                  │  │
│  │   Trainer   │────▶│  Rollout     │────▶│   Environment    │  │
│  │   (Policy   │     │  Workers     │     │   Pool           │  │
│  │   Update)   │◀────│  (Inference) │◀────│   (Sandboxes)    │  │
│  │             │     │              │     │                  │  │
│  └──────┬──────┘     └──────────────┘     └──────────────────┘  │
│         │                                                        │
│         │            ┌──────────────┐                            │
│         └───────────▶│  Reward      │                            │
│                      │  Models      │                            │
│                      └──────────────┘                            │
│                                                                   │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                     Ray Cluster                                   │
└──────────────────────────────────────────────────────────────────┘
```

The system has four major components:

1. **Trainer** — runs the policy gradient update on GPU nodes. Consumes rollout data, computes advantages, updates model weights.
2. **Rollout Workers** — run inference with the current policy to generate completions. Handle multi-turn conversations including tool calls.
3. **Environment Pool** — sandboxed execution environments (containers/microVMs) where tool calls are actually executed. Code runs here. Bash runs here. File operations happen here.
4. **Reward Models** — score completions along multiple dimensions (helpfulness, harmlessness, honesty). Separate from the policy model.

All of this runs on a **Ray cluster** that handles scheduling, resource allocation, fault detection, and communication between components.

---

## The Training Loop

One iteration of Composer 2's training loop:

```
1. Trainer broadcasts current policy weights
          │
          ▼
2. Rollout workers load weights, generate N completions
   for a batch of prompts
          │
          ▼
3. For each completion that includes tool calls:
   → Environment pool executes the tool
   → Result is fed back to the rollout worker
   → Worker continues generation (multi-turn)
          │
          ▼
4. Completed rollouts are scored by reward models
          │
          ▼
5. Trainer computes advantages, runs policy gradient
   update, produces new weights
          │
          ▼
6. Loop back to step 1
```

### Why this is hard

Each step has different compute characteristics:

| Step | Compute | Bottleneck |
|---|---|---|
| Policy update | GPU-bound | Memory (model size + optimizer state) |
| Rollout generation | GPU-bound | Throughput (many parallel completions) |
| Tool execution | CPU/IO-bound | Latency (sandboxed environments) |
| Reward scoring | GPU-bound | Throughput (batch scoring) |

Naive sequential execution leaves most resources idle most of the time. Composer 2's key architectural decision is to **overlap these phases** — rollout workers generate the next batch while the trainer updates on the current batch, and environment pools are pre-warmed and reused across steps.

---

## Ray Orchestration

Composer 2 uses [Ray](../systems/ray.md) as its distributed backbone. The choice is deliberate:

- **Heterogeneous resources.** Ray can schedule GPU tasks (training, inference) and CPU tasks (environment execution) on the same cluster, with fine-grained resource requirements per task.
- **Actor model.** Rollout workers and environments are long-lived Ray actors, avoiding the overhead of spinning up new processes per step.
- **Fault tolerance.** Ray's built-in actor supervision and task retry handle the inevitable failures in multi-day training runs.

### Resource layout

```
Ray Cluster
├── Head Node
│   └── Driver (orchestration logic)
├── GPU Nodes (N)
│   ├── Trainer Actor (multi-GPU, FSDP)
│   ├── Rollout Worker Actors (1 per GPU group)
│   └── Reward Model Actors
└── CPU Nodes (M)
    └── Environment Actors (many per node)
```

The driver script defines the training loop and dispatches work to actors. It doesn't do heavy computation itself — it's a coordinator.

### Scheduling subtleties

The interesting scheduling problem is **balancing rollout throughput with environment latency**. If a rollout involves 5 tool calls and each takes 2 seconds to execute, the rollout worker is blocked for 10 seconds — but only on CPU work. Composer 2 handles this by:

1. **Async tool execution.** Rollout workers submit tool calls to the environment pool and yield, allowing Ray to schedule other work on the same GPU.
2. **Batched continuation.** When tool results return, the rollout worker batches multiple continuations together for efficient GPU inference.
3. **Over-provisioning environments.** The environment pool has more capacity than the expected concurrent tool calls, absorbing variance in execution time.

---

## Rollout Workers

A rollout worker's job is straightforward: given a prompt and the current policy, generate a completion. What makes it complex is **multi-turn tool use**.

### Single-turn (simple case)

```
Prompt → [Policy Inference] → Completion → Done
```

### Multi-turn with tools

```
Prompt → [Policy Inference] → "Let me run this code..."
    → [Tool Call: execute_python]
    → [Environment: runs code, returns stdout]
    → [Policy Inference] → "The result is 42. Now let me..."
    → [Tool Call: execute_bash]
    → [Environment: runs bash, returns output]
    → [Policy Inference] → "Here's the final answer..."
    → Done
```

Each tool call is a round-trip to the environment pool. The rollout isn't complete until the model emits an end-of-turn token without a pending tool call.

### Implications for training

Multi-turn rollouts mean variable-length trajectories. Some rollouts finish in one turn; others take 10+ turns with complex tool interactions. This creates **load imbalance** — some rollout workers finish quickly while others are stuck on long multi-turn sequences.

Composer 2 addresses this with:
- **Work stealing.** Idle rollout workers can pick up prompts from a shared queue rather than waiting for their assigned batch.
- **Timeout policies.** Rollouts that exceed a maximum turn count or wall-clock time are truncated and scored as-is.
- **Stratified batching.** Prompts are grouped by expected complexity (estimated from prompt length and task type) to reduce variance within batches.

---

## Environment System

The environment pool is arguably the most novel part of Composer 2. When the model decides to call a tool, something has to **actually execute that tool** and return a result.

### Requirements

- **Isolation.** Code execution must be sandboxed. A model-generated `rm -rf /` should not affect anything outside the sandbox.
- **Low latency.** Environment startup must be fast (< 1 second). The model is waiting.
- **Statefulness.** Within a single rollout, the environment must maintain state across tool calls (files created in one call should exist in the next).
- **Reproducibility.** Given the same sequence of tool calls, the environment should produce the same outputs (important for debugging).
- **Scale.** Thousands of concurrent environments across the cluster.

### Implementation: microVM-based sandboxes

Composer 2 uses lightweight microVMs (similar in spirit to Firecracker / Anyrun):

```
┌────────────────────┐
│   Rollout Worker    │
│   (GPU node)        │
│                     │
│   tool_call ────────┼──────▶  ┌──────────────────┐
│                     │         │   microVM          │
│   result ◀──────────┼──────── │   ├── /tmp/work/   │
│                     │         │   ├── python3       │
│                     │         │   ├── bash          │
│                     │         │   └── [stateful     │
│                     │         │        filesystem]  │
└────────────────────┘         └──────────────────┘
```

Key design decisions:

- **Pre-warmed pool.** Environments are booted ahead of time and assigned to rollouts on demand. Boot time is amortized, not on the critical path.
- **Snapshot + restore.** Base environments are created from snapshots. State is preserved across tool calls within a rollout by keeping the VM alive. At rollout end, the VM is destroyed or reset from snapshot.
- **Resource limits.** Each environment has hard CPU, memory, and wall-clock limits. Infinite loops or memory bombs are killed cleanly.
- **Network isolation.** Environments cannot access the internet or the cluster network. They see only a loopback interface and mounted volumes.

### Interaction protocol

Communication between rollout workers and environments follows a simple request-response protocol:

```json
{
  "type": "execute_python",
  "code": "import math\nprint(math.sqrt(144))",
  "timeout_seconds": 30
}
→
{
  "stdout": "12.0\n",
  "stderr": "",
  "exit_code": 0,
  "execution_time_ms": 45
}
```

The protocol is intentionally minimal. Tool types (python, bash, file read/write) are a small fixed set. The model must produce valid tool calls that match the schema, and the RL training signal naturally pushes it toward correct tool use over time.

---

## Fault Tolerance

A Composer 2 training run lasts days to weeks. Hardware failures, OOM errors, network partitions, and environment hangs are **expected**, not exceptional.

### Failure modes and responses

| Failure | Detection | Response |
|---|---|---|
| Rollout worker crash | Ray actor death callback | Reassign prompts to surviving workers, respawn actor |
| Environment hang | Wall-clock timeout | Kill environment, return timeout error to rollout, model sees error as tool output |
| Trainer OOM | Exception in training loop | Reduce batch size or gradient accumulation, retry step |
| Network partition | Ray heartbeat timeout | Graceful degradation — continue with available workers, rebalance when healed |
| Reward model crash | Actor death callback | Respawn, re-score affected rollouts |

### Checkpointing strategy

See also: [Checkpointing](../systems/checkpointing.md)

Composer 2 checkpoints **asynchronously** — the trainer saves a snapshot of model weights + optimizer state to distributed storage while the next training step begins. This avoids blocking the training loop for checkpoint writes.

Checkpointing frequency is adaptive:
- **Time-based.** At least every N minutes (safety net).
- **Step-based.** Every K training steps (for reproducibility).
- **Quality-based.** When reward metrics improve significantly (to preserve good states).

On recovery, the system loads the latest valid checkpoint, recomputes any in-flight rollouts, and resumes. Rollouts are treated as disposable — losing a batch of rollouts is cheap compared to losing a training step.

---

## Key Insights

1. **RL post-training is an infrastructure problem.** The algorithmic core (policy gradients) is well-understood. The hard part is building a system that generates millions of multi-turn rollouts, executes tools safely, scores them, and trains — reliably, for weeks.

2. **The environment is a first-class component.** It's not an afterthought bolted onto the training loop. The quality and diversity of tool interactions directly shapes what the model learns. Poor environment design → poor tool use.

3. **Async everything.** The single biggest performance lever is overlapping phases — generating rollouts while training, executing tools while inferring, checkpointing while stepping. Sequential execution of the training loop leaves most resources idle.

4. **Fault tolerance through disposability.** Rather than making every component perfectly reliable, Composer 2 makes most components **disposable** and **restartable**. Rollouts can be regenerated. Environments can be reset. Only the model weights and optimizer state are precious.

5. **Ray's actor model is a natural fit.** Long-lived rollout workers, stateful environments, and heterogeneous resource requirements map cleanly onto Ray's actor and task abstractions. The alternative (custom distributed system) would be a massive engineering effort for marginal gains.

6. **Variable-length rollouts are the scheduling nightmare.** Multi-turn tool use means rollout times vary by 10-100x. Every design decision around batching, work stealing, and timeout policies exists to manage this variance.

---

*Next: [DeepSeek R1](deepseek-r1.md) for a different approach — large-scale GRPO without tool use, MoE architecture, and aggressive quantization.*
