# Communication Primitives

*Taxonomy — the collective operations that make multi-GPU training work.*

**TL;DR:** Everything in distributed deep learning reduces to a small set of group communication patterns, implemented by NCCL on top of NVLink / IB / RoCE. **All-reduce** sums tensors across ranks; **all-gather** collects shards into the whole; **reduce-scatter** sums and shards; **broadcast** sends one rank's tensor to all; **point-to-point** sends/receives between two ranks; **all-to-all** exchanges distinct payloads pairwise. Each primitive has an asymptotic bandwidth cost and an implementation choice (ring, tree, pipeline). Knowing the primitive tells you the communication cost of any parallelism scheme.

**Related taxonomies:** [_parallelism](../pre-training/_parallelism.md)
**Depth files covered here:** *(none yet — primitives described inline here; depth pages can be added for specific collectives if needed)*

---

## The problem

Every parallelism scheme ([data-parallelism](../pre-training/data-parallelism.md), [tensor-parallelism](../pre-training/tensor-parallelism.md), [pipeline-parallelism](../pre-training/pipeline-parallelism.md), etc.) is defined by **what gets split** and **what communication primitive closes the gap**. To evaluate scaling, understand bottlenecks, or tune a run, you need to know what the primitives are, what they cost, and how they're implemented.

Three levels of abstraction:

1. **Semantic level**: what does the collective produce? (E.g., all-reduce: every rank has the sum.)
2. **Algorithmic level**: how is it implemented? (E.g., ring all-reduce: reduce-scatter → all-gather.)
3. **Physical level**: what does the network actually carry? (E.g., 2·G bytes per rank over NVLink / IB).

This taxonomy covers all three.

---

## The shared pattern

Every collective operates on a **group** of N ranks and moves data between them. Three orthogonal axes:

- **Direction**: one-to-many (broadcast), many-to-one (gather), many-to-many (all-reduce, all-gather, all-to-all).
- **Operation**: identity (gather), sum / max / min (reduce).
- **Result distribution**: same-on-every-rank (all-reduce, all-gather) vs different-on-each-rank (reduce-scatter, all-to-all).

The matrix of combinations:

| Collective | Direction | Reduction | Every rank same output? |
|---|---|---|---|
| Broadcast | 1 → N | None | Yes |
| Reduce | N → 1 | Yes | No (only root) |
| All-reduce | N ↔ N | Yes | **Yes** |
| Scatter | 1 → N | None | No (shard per rank) |
| Gather | N → 1 | None | No (only root) |
| All-gather | N ↔ N | None | **Yes** (full gather on all) |
| Reduce-scatter | N ↔ N | Yes | **No** (shard of the sum per rank) |
| All-to-all | N ↔ N | None | **No** (per-pair different payloads) |
| Point-to-point (Send/Recv) | 1 → 1 | None | N/A |

---

## The primitives, in depth

### 1. All-reduce

**Semantic**: every rank contributes a tensor `x_i`; every rank ends up with `Σ_i x_i` (or max / min / other reduction). Identical result on every rank.

```
Before:  rank 0: [1 2 3]   rank 1: [4 5 6]   rank 2: [7 8 9]
After:   rank 0: [12 15 18] rank 1: [12 15 18] rank 2: [12 15 18]
```

**Used by**: [data-parallelism](../pre-training/data-parallelism.md) (gradient sync), [tensor-parallelism](../pre-training/tensor-parallelism.md) (partial-sum combine).

**Algorithm — ring all-reduce**: the standard implementation. Decompose into:

```
all-reduce = reduce-scatter + all-gather
```

Both phases use a ring topology. For N ranks, tensor size G:

- **Reduce-scatter phase**: N − 1 steps. In step k, each rank sends one chunk (size G/N) to its right neighbor and receives one chunk from its left, accumulating into its own slot. After N − 1 steps, each rank holds the reduced value of one chunk.
- **All-gather phase**: N − 1 steps. Each rank rotates its held chunk around the ring; after N − 1 steps, every rank has every chunk.

Bandwidth per rank: `2 · G · (N − 1) / N ≈ 2G` at large N.

```
Diagram: Ring all-reduce of tensor [A B C] across 3 ranks (ranks hold [a1 b1 c1], [a2 b2 c2], [a3 b3 c3])

Step 0 (Reduce-scatter):
  rank 0: send c1 → rank 1; recv b3 ← rank 2 → holds b = b1+b3 (accumulating)
  rank 1: send a2 → rank 2; recv c1 ← rank 0 → holds c = c1+c2
  rank 2: send b3 → rank 0; recv a2 ← rank 1 → holds a = a2+a3

Step 1 (Reduce-scatter):
  rank 0: send (b1+b3) → rank 1; recv (a2+a3) ← rank 2 → holds a = a1+a2+a3 ✓
  rank 1: send (c1+c2) → rank 2; recv (b1+b3) ← rank 0 → holds b = b1+b2+b3 ✓
  rank 2: send (a2+a3) → rank 0; recv (c1+c2) ← rank 1 → holds c = c1+c2+c3 ✓

Now each rank has 1/3 of the fully-reduced tensor. Now all-gather phase:

Step 2 (All-gather):
  rank 0: send a → rank 1; recv c ← rank 2 → holds [a, ?, c]
  rank 1: send b → rank 2; recv a ← rank 0 → holds [a, b, ?]
  rank 2: send c → rank 0; recv b ← rank 1 → holds [?, b, c]

Step 3 (All-gather):
  rank 0: send c → rank 1; recv b ← rank 2 → holds [a, b, c] ✓
  rank 1: send a → rank 2; recv c ← rank 0 → holds [a, b, c] ✓
  rank 2: send b → rank 0; recv a ← rank 1 → holds [a, b, c] ✓
```

Bandwidth optimal: each rank sends `2G(N−1)/N` bytes — provably optimal for all-reduce.

**Algorithm — tree all-reduce**: at very large N with low-latency fabric, a binary-tree reduce + broadcast can have lower *latency* (log N steps vs 2N − 2 for ring) at the cost of bandwidth. NCCL auto-selects ring vs tree based on topology.

**NCCL implementation**: `ncclAllReduce()`.

### 2. All-gather

**Semantic**: each rank contributes a shard `x_i`; every rank ends up with the concatenation `[x_0; x_1; ...; x_{N-1}]`.

```
Before:  rank 0: [a]   rank 1: [b]   rank 2: [c]
After:   rank 0: [a b c]   rank 1: [a b c]   rank 2: [a b c]
```

**Used by**: [fsdp](../pre-training/fsdp.md) (param reconstruction), [context-parallelism](../pre-training/context-parallelism.md) (K/V gather), [sequence-parallelism](../pre-training/sequence-parallelism.md) (g operator forward).

**Algorithm — ring all-gather**: N − 1 steps of rotation. Each rank sends its current chunk to the right, receives from the left; after N − 1 steps all ranks hold all chunks.

Bandwidth per rank: `G · (N − 1) / N ≈ G`.

**NCCL implementation**: `ncclAllGather()`.

### 3. Reduce-scatter

**Semantic**: each rank contributes a tensor `x_i`; the summed tensor `Σ_i x_i` is split into N chunks, and rank r receives chunk r.

```
Before:  rank 0: [a1 b1 c1]   rank 1: [a2 b2 c2]   rank 2: [a3 b3 c3]
After:   rank 0: [a1+a2+a3]   rank 1: [b1+b2+b3]   rank 2: [c1+c2+c3]
```

**Used by**: [fsdp](../pre-training/fsdp.md) (gradient shard-reduction), [sequence-parallelism](../pre-training/sequence-parallelism.md) (ḡ operator forward).

**Algorithm — ring reduce-scatter**: N − 1 steps; same ring as all-reduce's first phase.

Bandwidth per rank: `G · (N − 1) / N ≈ G`.

**NCCL implementation**: `ncclReduceScatter()`.

### 4. Broadcast

**Semantic**: one rank's tensor is copied to all other ranks in the group.

```
Before:  rank 0 (root): [a b c]   rank 1: [?]   rank 2: [?]
After:   rank 0: [a b c]   rank 1: [a b c]   rank 2: [a b c]
```

**Used by**: model initialization (rank 0 broadcasts weights to all), parameter server schemes.

**Algorithm — tree broadcast**: log N steps. Root sends to 2 children, each of whom sends to their children, and so on.

**Bandwidth per rank** (per-hop): `G`. Total data movement: `O(G · log N)` across the tree.

**NCCL implementation**: `ncclBroadcast()`.

### 5. Reduce

**Semantic**: every rank contributes `x_i`; the root rank ends up with `Σ_i x_i`. Non-root ranks are done.

Less common in modern deep-learning — usually you want `all-reduce` so every rank has the result. Used in some gradient aggregation schemes for parameter-server topologies.

**NCCL implementation**: `ncclReduce()`.

### 6. All-to-all

**Semantic**: each rank has N payloads, one for each destination rank. After all-to-all, each rank has N payloads received from every other rank.

```
Before:
  rank 0: [to_0=α, to_1=β, to_2=γ]
  rank 1: [to_0=δ, to_1=ε, to_2=ζ]
  rank 2: [to_0=η, to_1=θ, to_2=ι]

After:
  rank 0: [from_0=α, from_1=δ, from_2=η]
  rank 1: [from_0=β, from_1=ε, from_2=θ]
  rank 2: [from_0=γ, from_1=ζ, from_2=ι]
```

Think of it as "transpose across ranks" — the (i, j) position becomes (j, i).

**Used by**: [expert-parallelism](../pre-training/expert-parallelism.md) (MoE token dispatch + combine), alltoall-based attention schemes.

**Algorithm**: naïve pairwise (each rank sends to every other rank sequentially) is `N − 1` steps with full bandwidth. Sophisticated algorithms (Bruck, pairwise-exchange) can reduce latency.

Bandwidth per rank: `G · (N − 1) / N ≈ G` where G is each rank's total send volume.

**Variants**:
- **all-to-all single (uniform)**: every rank sends equal-sized chunks. Simple.
- **all-to-all variable (unequal)**: each rank sends/receives different sized payloads. Critical for MoE with imbalanced expert loads.

**NCCL implementation**: `ncclSend()`/`ncclRecv()` pairs with grouped launches, or higher-level wrappers in PyTorch `dist.all_to_all_single()`.

### 7. Point-to-point (Send / Recv)

**Semantic**: rank A sends a tensor to rank B. No reduction, no fan-out.

**Used by**: [pipeline-parallelism](../pre-training/pipeline-parallelism.md) (activation/gradient boundary transfers), RL-rollout dispatch.

Cheapest primitive: only two ranks involved, no group-wide collective overhead.

**NCCL implementation**: `ncclSend()`, `ncclRecv()`.

---

## Bandwidth cost summary

For a group of N ranks exchanging a tensor of size G bytes per rank:

| Primitive | Bandwidth per rank | Notes |
|---|---|---|
| Broadcast | ≈ G | Tree-based |
| Reduce | ≈ G | Root receives, others send |
| **All-reduce** | **≈ 2G** | Ring: reduce-scatter (G) + all-gather (G) |
| All-gather | ≈ G | Ring rotation |
| Reduce-scatter | ≈ G | Ring rotation with accumulation |
| All-to-all | ≈ G | Every rank exchanges ~G total |
| Send / Recv | ≤ G | Only two ranks |

Note: all-reduce is **twice the cost** of all-gather or reduce-scatter. This is why Megatron-SP is bandwidth-equivalent to Megatron-TP (it replaces one all-reduce with one all-gather + one reduce-scatter).

---

## Physical layer

### NVLink (intra-server)

- 8 GPUs per DGX server, all NVLink-connected via NVSwitch.
- H100: 900 GB/s per GPU (18 × 50 GB/s NVLink ports).
- B200: 1.8 TB/s per GPU.
- Latency: sub-microsecond.
- Why TP stays intra-server: the all-reduce in attention/MLP happens N times per layer per step; only NVLink bandwidth keeps it competitive with compute.

### InfiniBand (IB) / RoCE (inter-server)

- 400 Gbps per GPU (NDR-400) = 50 GB/s.
- RoCE = RDMA over Converged Ethernet; IB = InfiniBand.
- Llama 3 chose RoCE for 405B (Section 3.3.1); smaller models used IB.
- Latency: 1–10 microseconds typical, depending on topology.
- Why DP scales cross-server: the gradient all-reduce happens once per step, with bandwidth ≈ 2·params. 50 GB/s absorbs this for most models.

### Topology awareness

Modern clusters are hierarchical:
- Within rack: NVLink (intra-node) + NVSwitch + maybe local ToR.
- Across racks: aggregation layer with oversubscription (Llama 3: 1:7).

NCCL picks collective algorithms based on topology: ring within NVLink domain, tree across IB, hybrid for very large all-reduces.

### Hybrid collective algorithms

At 10K+ GPUs, flat ring or flat tree is inefficient. **Hierarchical all-reduce**: reduce-scatter within nodes → small all-reduce across nodes → all-gather within nodes. Reduces inter-node traffic.

NCCL's `NCCL_ALGO` environment variable exposes this: `Ring`, `Tree`, `CollnetDirect`, `CollnetChain`, `NVLS` (new in Hopper-era NCCL).

---

## When to use which

The decision is made by your parallelism scheme, not by you directly. But understanding the primitives helps debug:

| You see in a profiler | It means |
|---|---|
| Large `ncclAllReduce` on gradients | DDP's gradient sync (or Megatron TP's all-reduce) |
| Lots of `ncclAllGather` + `ncclReduceScatter` | FSDP / ZeRO-3 in flight |
| `ncclAllReduce` inside forward (many per layer) | Megatron TP |
| `ncclAllGather` inside attention | Context parallelism (all-gather variant) |
| `ncclSend/Recv` at regular intervals | Pipeline parallelism micro-batches |
| Many `ncclSend/Recv` inside a single layer | MoE all-to-all (implemented as grouped Send/Recv) |

---

## Primitive-level diagrams

### All-reduce as reduce-scatter + all-gather

```
Phase 1 — Reduce-scatter (N=4, tensor split into 4 chunks):

Step 0:   Each rank sends chunk (rank) to right, receives chunk (rank-1) from left:
  r0 --[c0]--> r1
  r1 --[c1]--> r2
  r2 --[c2]--> r3
  r3 --[c3]--> r0

After receive, each rank accumulates.

After N-1 = 3 steps of rotate-and-accumulate:
  r0 holds: full sum of chunk 3
  r1 holds: full sum of chunk 0
  r2 holds: full sum of chunk 1
  r3 holds: full sum of chunk 2

Phase 2 — All-gather:
Step 0..2: Each rank rotates its fully-reduced chunk; after N-1 steps all ranks hold all chunks.

Result: every rank has the full reduced tensor.
```

### All-to-all visualization

```
Before (each rank has N payloads, one per dest):
  rank 0: [α → 0, β → 1, γ → 2, δ → 3]
  rank 1: [ε → 0, ζ → 1, η → 2, θ → 3]
  rank 2: [ι → 0, κ → 1, λ → 2, μ → 3]
  rank 3: [ν → 0, ξ → 1, ο → 2, π → 3]

After (each rank has N payloads, one from each source):
  rank 0: [α ← 0, ε ← 1, ι ← 2, ν ← 3]
  rank 1: [β ← 0, ζ ← 1, κ ← 2, ξ ← 3]
  rank 2: [γ ← 0, η ← 1, λ ← 2, ο ← 3]
  rank 3: [δ ← 0, θ ← 1, μ ← 2, π ← 3]

Each pair (i, j) exchanges distinct payloads — essentially a matrix transpose across ranks.
```

---

## Sources

- Paper: *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations* — Patarasuk & Yuan, 2009 — the ring algorithm and its bandwidth-optimality proof.
- Paper: *Horovod: fast and easy distributed deep learning in TensorFlow* — Sergeev & Del Balso, 2018, arXiv 1802.05799 — canonical ring all-reduce in DL.
- Docs: NVIDIA NCCL — https://docs.nvidia.com/deeplearning/nccl/.
- Docs: PyTorch distributed — https://pytorch.org/docs/stable/distributed.html.
- Paper: *Pregel: A System for Large-Scale Graph Processing* — Malewicz et al., 2010 — the BSP communication model that underlies collective thinking.
