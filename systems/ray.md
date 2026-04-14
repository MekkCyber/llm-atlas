# Ray

*The distributed execution framework that underpins most modern AI training systems — and the mental model you need to reason about it.*

**Used in:** [Composer 2](../case-studies/composer2.md) · [Rollout systems](../post-training/rollout-systems.md)

---

## What Ray Is

Ray is a general-purpose distributed computing framework. It lets you take Python functions and classes and run them across a cluster of machines, handling scheduling, communication, fault detection, and resource management.

In the context of AI training, Ray serves as the **orchestration layer** — the thing that decides which code runs on which machine, when, and with what resources. It's used by most major RL post-training systems because it solves a specific problem well: **heterogeneous, dynamic workloads** where some tasks need GPUs, some need CPUs, some are short-lived, and some are long-lived.

---

## Mental Model

The key abstraction in Ray is the distinction between **tasks** and **actors**.

### Tasks: stateless, short-lived

A Ray task is a function invocation scheduled on a remote worker:

```python
@ray.remote
def score_completion(completion: str) -> float:
    # runs on whatever worker Ray assigns
    return reward_model.score(completion)

# launches 1000 scoring tasks in parallel
futures = [score_completion.remote(c) for c in completions]
scores = ray.get(futures)  # blocks until all complete
```

Tasks are **stateless** — each invocation is independent. Ray can schedule them on any available worker, retry them on failure, and load-balance across the cluster.

**When to use tasks:** embarrassingly parallel work, pure functions, things you'd put in a thread pool.

### Actors: stateful, long-lived

A Ray actor is an instance of a class running on a remote worker:

```python
@ray.remote(num_gpus=1)
class RolloutWorker:
    def __init__(self, model_config):
        self.model = load_model(model_config)  # lives for the actor's lifetime
    
    def generate(self, prompt: str) -> str:
        return self.model.generate(prompt)
    
    def update_weights(self, new_weights):
        self.model.load_state_dict(new_weights)

# create 8 workers, each pinned to a GPU
workers = [RolloutWorker.remote(config) for _ in range(8)]
```

Actors are **stateful** — they hold data in memory (like loaded model weights) and can be called repeatedly. They're pinned to a specific process and machine for their lifetime.

**When to use actors:** anything with expensive initialization (model loading), mutable state (environment VMs), or long-lived sessions (rollout workers).

### The driver

The driver is your main Python script. It's where you define the computation graph — creating actors, submitting tasks, collecting results, and coordinating the overall flow:

```python
# driver script — runs on the head node
trainer = Trainer.remote(model_config)
workers = [RolloutWorker.remote(config) for _ in range(8)]
envs = [Environment.remote() for _ in range(64)]

for step in range(num_steps):
    weights = ray.get(trainer.get_weights.remote())
    for w in workers:
        w.update_weights.remote(weights)
    
    rollouts = ray.get([w.generate.remote(prompts[i]) for i, w in enumerate(workers)])
    scores = ray.get([score_completion.remote(r) for r in rollouts])
    trainer.train_step.remote(rollouts, scores)
```

The driver does almost no compute itself. It's a **coordinator** — dispatching work and collecting results.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Ray Cluster                  │
│                                               │
│  ┌──────────┐                                │
│  │  Head     │  GCS (Global Control Store)    │
│  │  Node     │  ├── Actor registry            │
│  │          │  ├── Resource table             │
│  │  Driver ──┤  └── Object directory           │
│  │          │                                 │
│  └──────────┘  Scheduler                      │
│       │        ├── Placement decisions         │
│       │        └── Resource matching           │
│       │                                       │
│  ┌────┴──────────────────────────────────┐   │
│  │        Worker Nodes                     │   │
│  │                                         │   │
│  │  ┌─────────┐  ┌─────────┐             │   │
│  │  │Worker 1 │  │Worker 2 │  ...         │   │
│  │  │(2 GPU)  │  │(CPU)    │             │   │
│  │  │         │  │         │             │   │
│  │  │Actor A  │  │Actor B  │             │   │
│  │  │Actor C  │  │Task pool│             │   │
│  │  └─────────┘  └─────────┘             │   │
│  └────────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

Key components:

- **Global Control Store (GCS):** Metadata service that tracks what actors exist, where they're running, and what resources are available. Runs on the head node.
- **Scheduler:** Decides where to place tasks and actors based on resource requirements (`num_gpus`, `num_cpus`, custom resources) and data locality.
- **Object Store:** Shared-memory system (built on Apache Arrow / Plasma) that allows zero-copy data sharing between tasks/actors on the same node. Cross-node transfers go over the network.
- **Raylet:** Per-node daemon that manages local workers, object store, and communicates with GCS.

---

## How Scheduling Works

When you call `actor.method.remote()` or `function.remote()`, Ray:

1. Serializes the arguments
2. Looks up resource requirements (from `@ray.remote` decorator)
3. Finds a node with sufficient available resources
4. Ships the serialized task to that node's raylet
5. Raylet assigns it to a local worker process
6. Worker executes, serializes the return value
7. Return value is stored in the local object store
8. Caller can retrieve it with `ray.get()`

### Resource matching

Resources are **logical labels**, not physical constraints:

```python
@ray.remote(num_gpus=2, num_cpus=4)
class Trainer:
    ...

@ray.remote(num_cpus=1)
class Environment:
    ...
```

Ray won't schedule a `Trainer` on a node with only 1 GPU. It will pack as many `Environment` actors onto a CPU node as the CPU count allows.

You can define **custom resources** for fine-grained control:

```python
@ray.remote(resources={"environment_slots": 1})
class Environment:
    ...

# Node configured with {"environment_slots": 16}
# → at most 16 environments per node
```

### Placement groups

For workloads that need co-located resources (e.g., FSDP training across 8 GPUs on the same node), Ray provides placement groups:

```python
pg = placement_group([{"GPU": 1}] * 8, strategy="PACK")
ray.get(pg.ready())

# All 8 actors land on the same node (or fewest nodes possible)
workers = [Worker.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(pg, bundle_index=i)
).remote() for i in range(8)]
```

---

## How It's Used in RL Systems

In a system like [Composer 2](../case-studies/composer2.md), Ray's role is to manage four distinct workloads on a shared cluster:

| Workload | Ray abstraction | Resources |
|---|---|---|
| Policy training | Actor (long-lived, stateful) | Multi-GPU per actor |
| Rollout generation | Actor (long-lived, stateful) | 1+ GPU per actor |
| Reward scoring | Actor or Task | GPU, can share with rollout |
| Tool execution | Actor (stateful per-rollout) | CPU only |

The scheduling challenge is that these workloads have different and competing resource needs:

- Training wants all GPUs for gradient computation
- Rollouts want GPUs for inference, but can release them during tool execution
- Environments want CPUs and memory, but no GPUs
- Reward models want GPUs for batch scoring

Ray handles this through **resource multiplexing** — actors declare their resource needs, and the scheduler ensures total allocated resources don't exceed physical capacity. When a rollout worker is blocked waiting for tool execution, its GPU is technically "claimed" but idle, which is why systems like Composer 2 use [async tool execution](../case-studies/composer2.md#scheduling-subtleties) to yield the GPU during waits.

---

## Common Pitfalls

**1. Serialization overhead.** Ray serializes arguments and return values with `pickle` (or Apache Arrow for numpy arrays). Passing large objects as task arguments instead of using the object store causes repeated serialization.

```python
# bad — serializes big_data N times
futures = [process.remote(big_data) for _ in range(N)]

# good — serialize once, pass reference
ref = ray.put(big_data)
futures = [process.remote(ref) for _ in range(N)]
```

**2. Head-of-line blocking.** If you `ray.get()` on a single slow task, the driver blocks even if other results are ready. Use `ray.wait()` for incremental processing:

```python
# bad — blocks until ALL tasks complete
results = ray.get(futures)

# good — process results as they arrive
pending = list(futures)
while pending:
    ready, pending = ray.wait(pending, num_returns=1)
    result = ray.get(ready[0])
    process(result)
```

**3. Actor death is silent by default.** If an actor crashes, calls to it will raise an exception — but only when you `ray.get()` the result. For critical actors (trainers, workers), use `max_restarts` and health checks:

```python
@ray.remote(max_restarts=3, max_task_retries=2)
class CriticalWorker:
    ...
```

**4. Memory leaks in the object store.** Objects in Ray's object store are reference-counted. If you hold references (futures) to many large objects without releasing them, the object store fills up and spills to disk, killing performance.

---

## Key Takeaway

Ray's power in post-training comes from three properties:

1. **Heterogeneous resource management** — GPUs and CPUs on the same cluster, with fine-grained allocation
2. **Actor model** — long-lived stateful processes (models, environments) without per-call overhead
3. **Flexible scheduling** — the driver defines the computation graph dynamically, adapting to failures and load imbalance

The mental model: Ray is a **programmable cluster scheduler** that speaks Python. You describe what needs to run and what resources it needs; Ray figures out where and when.

---

*See also: [Fault tolerance](fault-tolerance.md) · [Composer 2 — Ray orchestration](../case-studies/composer2.md#ray-orchestration)*
