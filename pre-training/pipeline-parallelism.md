# Pipeline Parallelism (PP)

*Depth — split the model into contiguous layer-groups (stages) across GPUs; flow micro-batches through the pipeline.*

**TL;DR:** Put layers 1–8 on GPU 1, layers 9–16 on GPU 2, etc. Feed the forward pass through like an assembly line. The problem: the first stage finishes its micro-batch and then **idles while later stages are still working** — the "pipeline bubble." Mitigation: use many micro-batches (GPipe) or interleave forward/backward passes (1F1B, PipeDream) to keep every stage busy. **Interleaved 1F1B** (Megatron-2) spreads `V` virtual stages per rank throughout the model so bubbles shrink by `V`×. Communication: only point-to-point activation/gradient transfer at stage boundaries — cheap per transfer. Pipeline bubbles, not comm, are the real cost.

**Prereqs:** [_parallelism](_parallelism.md), [_communication-primitives](../systems/_communication-primitives.md)
**Related:** [fsdp](fsdp.md) · [tensor-parallelism](tensor-parallelism.md) · [dualpipe](../systems/dualpipe.md)

---

## What it is

Model has L layers. Split into P contiguous **stages**:
- Stage 0: layers 0 to L/P − 1 on GPU 0.
- Stage 1: layers L/P to 2L/P − 1 on GPU 1.
- ...
- Stage P − 1: layers (P − 1)L/P to L − 1 on GPU P − 1.

A full forward pass on a global batch B:
1. Feed B into Stage 0. It produces activations `a_0 → ... → a_{L/P}`.
2. **Transfer the final activation of Stage 0 to Stage 1** (point-to-point).
3. Stage 1 continues, passes to Stage 2, and so on.
4. Stage P − 1 computes the final logits and loss.

The backward pass reverses this. Loss gradient flows back stage-by-stage, with point-to-point transfers at each boundary.

Communication is tiny: only the activation tensor at each stage boundary (`B · S · H` bytes). The problem is *idle time*.

### The pipeline bubble

Naïve pipeline: feed *the whole batch* through Stage 0, then Stage 1, then... the later stages wait for earlier stages to finish. Terrible utilization.

**Standard fix**: split the batch into M micro-batches of size `B/M`. Feed micro-batches through in a pipelined fashion:

```
Time →
Stage 0:  [FM1][FM2][FM3][FM4] ... idle    ← forward done, waiting for backward
Stage 1:       [FM1][FM2][FM3][FM4] ... idle
Stage 2:            [FM1][FM2][FM3][FM4] ... idle
Stage 3:                 [FM1][FM2][FM3][FM4] ... idle
                                           [BM4][BM3][BM2][BM1]   ← backward
Stage 2:                              ... [BM4][BM3][BM2][BM1]
Stage 1:                         ... [BM4][BM3][BM2][BM1]
Stage 0:                    ... [BM4][BM3][BM2][BM1]
```

The gaps at the start and end are the **pipeline bubble**.

### GPipe's bubble formula

For P stages and M micro-batches, the fraction of total time wasted in the bubble (Huang 2019):

```
bubble_fraction = (P − 1) / (M + P − 1)
```

- P = 4, M = 4 → 3/7 ≈ 43% wasted.
- P = 4, M = 16 → 3/19 ≈ 16% wasted.
- P = 16, M = 4 → 15/19 ≈ 79% wasted — almost no work done.

The trend: **you want M ≫ P** for the bubble to be small. GPipe's empirical rule: **M ≥ 4 · P**.

But M is bounded by activation memory. All M micro-batches' activations must live in each stage until backward — if M is too large, you OOM. GPipe mitigates with **activation checkpointing** (discard activations in forward, recompute in backward), trading compute for memory.

---

## Schedules

### GPipe (flush-at-end)

Forward all M micro-batches, then backward all M. Simple. Synchronous SGD — one weight update per batch.

- Bubble: `(P − 1) / (M + P − 1)`.
- Activation memory per stage: all M micro-batches' activations.
- Weight updates: synchronous.

### PipeDream 1F1B (one-forward-one-backward)

Interleave forward and backward passes. In steady state, each stage alternates:

```
Stage i:  Fᵢ Fᵢ Fᵢ Fᵢ [startup]   then   Fᵢ Bᵢ Fᵢ Bᵢ Fᵢ Bᵢ ... Bᵢ Bᵢ Bᵢ [drain]
```

After the startup phase (filling the pipeline), each stage does F, B, F, B, ... The activation memory is now only **O(depth of pipeline)** micro-batches per stage (instead of M), because earlier-processed micro-batches' backwards happen before later micro-batches' forwards.

Same bubble formula as GPipe (asymptotically), but **much less activation memory**. PipeDream's original form was asynchronous (different stages update with different weight versions), requiring weight stashing to keep one weight copy per in-flight micro-batch. Modern "PipeDream-Flush" / "1F1B-synchronous" keeps synchronous SGD but adopts the 1F1B schedule.

### Interleaved 1F1B (Megatron-2)

Instead of one contiguous stage per rank, each rank holds **V virtual stages / chunks** interleaved through the model. For V=2: rank 0 holds layers 0–3 AND layers 16–19; rank 1 holds layers 4–7 AND layers 20–23; etc.

Forward/backward follow a micro-schedule that visits all V chunks per rank per micro-batch. Bubble formula (Narayanan 2021):

```
bubble_fraction_interleaved = (P − 1) / (V · M)
```

— shrinks by V× compared to non-interleaved. Cost: V× more point-to-point transfers per micro-batch (same per-transfer volume).

- V = 1: standard 1F1B.
- V = 4: common in Megatron-2's LLM runs. Bubble shrinks 4×.

Constraint: M must be divisible by P for scheduling. Not all architectures cleanly split into V chunks of equal compute.

### Llama 3's modified interleaved

Llama 3 (Sec. 3.3.2) adds two tweaks on top of interleaved 1F1B:

1. **Unbalanced first/last stages.** The first stage holds the embedding; the last holds the output projection + loss. These have less compute than an interior Transformer block. Llama 3 **removes 1 Transformer layer from each end stage** to rebalance.
2. **Tunable N (micro-batch-group size).** Independent of P and M. Picks a sweet spot between depth-first (DFS) and breadth-first (BFS) schedules.

Llama 3 reports training 8K-context sequences **without activation checkpointing** thanks to these PP optimizations.

---

## How it works

### Communication primitive: point-to-point

Between stage i and stage i + 1, the activation tensor is sent from GPU i to GPU i + 1 during forward; the gradient tensor flows the other way during backward. NCCL `Send/Recv`.

```python
# Forward boundary: stage i sends activations to stage i+1
if stage == i:
    torch.distributed.send(activation, dst=stage_i_plus_1_rank)
elif stage == i + 1:
    torch.distributed.recv(activation, src=stage_i_rank)

# Backward boundary: stage i+1 sends grad to stage i
if stage == i + 1:
    torch.distributed.send(grad, dst=stage_i_rank)
elif stage == i:
    torch.distributed.recv(grad, src=stage_i_plus_1_rank)
```

Cost per transfer: `B_micro · S · H` bytes. A full step (M micro-batches, P − 1 boundaries, forward + backward) → `2 · M · (P − 1) · B_micro · S · H` bytes total across the pipeline, but each rank only sends/receives a small subset.

Point-to-point is **much cheaper** than all-reduce — only two GPUs involved per transfer. This is why PP scales across servers while TP doesn't.

### The forward pass, interleaved

```python
# Pseudocode for one rank in 1F1B
def stage_loop(stage_id, P, M):
    num_warmup = P - stage_id - 1     # number of forward-only micro-batches before 1F1B
    num_1f1b = M - num_warmup

    # Warmup: just forward
    for m in range(num_warmup):
        act_in = recv_from_prev()        # nothing to recv for stage 0
        act_out = forward(local_layers, act_in, micro_batch=m)
        send_to_next(act_out)
        store_for_backward(act_out)

    # Steady state: 1 forward, 1 backward
    for m in range(num_1f1b):
        # forward new micro-batch
        act_in = recv_from_prev()
        act_out = forward(local_layers, act_in, micro_batch=num_warmup + m)
        send_to_next(act_out)
        store_for_backward(act_out)

        # backward old micro-batch
        grad_out = recv_from_next()
        grad_in = backward(local_layers, grad_out, micro_batch=m)
        send_to_prev(grad_in)

    # Cool-down: just backward
    for m in range(num_warmup):
        grad_out = recv_from_next()
        grad_in = backward(local_layers, grad_out, micro_batch=num_1f1b + m)
        send_to_prev(grad_in)

    all_reduce(local_gradients)  # if DP is present, only at end
    optimizer.step()
```

### The backward pass

Each stage holds activations from its stored forward passes. Backward reuses them to compute local gradients, propagates grad-input back via point-to-point.

### Activation memory

In 1F1B:
- Stage 0 peaks at `P` in-flight micro-batches (worst case).
- Stage P − 1 peaks at 1 in-flight micro-batch.
- Average: `P/2` micro-batches per stage.

Much better than GPipe's `M` per stage.

With interleaved (V chunks/rank) and selective recompute: peak activation memory is `~V · P / (V·M)` fraction of full-batch activations. Sec. 4 of Korthikanti 2022 shows the memory math in detail.

---

## Communication vs bubble tradeoff

PP's scaling story:

| Variable | Effect on bubble | Effect on memory | Effect on comm |
|---|---|---|---|
| P (stages) ↑ | Bubble grows | Less params/stage | More p2p transfers |
| M (micro-batches) ↑ | Bubble shrinks | More activations | More p2p, smaller each |
| V (interleave) ↑ | Bubble shrinks V× | Negligible | V× more p2p |

Typical production settings: **P = 4–16**, **M = 4·P–16·P**, **V = 1–4**.

Llama 3 at 405B: P = 16, V = 2, M chosen for each of the Table 4 configurations (see [_parallelism](_parallelism.md)).

---

## Why it matters

- **Memory scaling across servers.** PP is the only parallelism axis that cleanly scales model size across servers (since TP is intra-server). A trillion-parameter dense model needs PP.
- **Cheap comm.** Only point-to-point, no all-reduces. Works on RoCE/IB at 400 Gbps.
- **Composes with TP.** Each PP stage can be TP-sharded internally. Megatron-2's 3D parallelism is exactly TP × PP × DP.
- **Bubbles cap efficiency.** Even interleaved 1F1B gives up 5–15% throughput to bubbles. For wallclock-critical runs, **DualPipe** (DeepSeek-V3, [systems/dualpipe](../systems/dualpipe.md)) cuts this further with bidirectional scheduling.
- **The right tool when TP + FSDP aren't enough.** At 500B+ dense, FSDP's all-gather cost + TP's NVLink cap leave PP as the remaining scaling axis.

---

## Gotchas & tricks

- **M ≥ 4P** or the bubble eats everything. Tune M as high as memory allows.
- **Activation checkpointing is frequently mandatory.** Without it, M large enough to hide the bubble blows activation memory. With it, you pay recompute overhead in backward. Selective recompute (Korthikanti 2022) — checkpoint only specific layers — gives the best trade-off.
- **Load balance is surprisingly hard.** First stage has embedding; last has output projection + loss. Both have less compute than interior blocks. Llama 3 rebalances by moving 1 layer off each end.
- **PP + TP composition.** PP across servers, TP within a server. DP (or FSDP) wraps the whole thing. Order in the topology mesh: `[TP, PP, DP]` or `[TP, CP, PP, DP]` with TP innermost (fastest links).
- **Micro-batch size matters.** Too small → poor per-stage GPU utilization (kernel launch overhead, small GEMMs). Too large → memory pressure. Sweet spot typically `B_micro ∈ [1, 4]` at frontier scale.
- **1F1B vs GPipe for memory.** 1F1B is strictly better for memory. No reason to use GPipe in new code.
- **Weight stashing (PipeDream-async) is rarely used at scale.** Modern runs use 1F1B-synchronous (or "PipeDream-Flush"), which is essentially GPipe's synchrony with 1F1B's memory pattern.
- **Cross-boundary recomputation.** If you checkpoint at the end of each PP stage, you recompute forward for backward at every boundary. Expensive.
- **Optimizer state lives with the stage.** Stage i has its own optimizer states for its layers. Adam state does not cross stage boundaries.
- **Pipeline bubble debugging.** Use `torch.profiler` + nvidia-smi to see per-GPU utilization. If one stage is consistently idle, rebalance.
- **PP + FSDP tension.** FSDP all-gathers per-layer params. If a PP stage has few layers, FSDP's per-unit cost dominates. Pair FSDP-per-stage with sensible wrap granularity; or use FSDP-over-DP-only (don't FSDP the PP stages' params).
- **Loss parallelism.** Only the last stage computes loss. The loss-gather is usually cheap but must be done correctly for metrics.
- **DualPipe further improvement.** DeepSeek-V3's DualPipe (see [systems/dualpipe](../systems/dualpipe.md)) runs forward and backward **bidirectionally** so the pipeline has fewer dead cycles. Halves the bubble at the cost of bookkeeping complexity.

---

## Sources

- Paper: *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* — Huang et al., NeurIPS 2019, arXiv 1811.06965 — the bubble formula, activation checkpointing trick.
- Paper: *PipeDream: Fast and Efficient Pipeline Parallel DNN Training* — Harlap et al., 2018, arXiv 1806.03377 — 1F1B schedule.
- Paper: *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* — Narayanan et al., 2021, arXiv 2104.04473 — interleaved 1F1B, 3D parallelism.
- Paper: *Reducing Activation Recomputation in Large Transformer Models* — Korthikanti et al., 2022, arXiv 2205.05198 — selective recompute + memory analysis.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — modified interleaved with tunable N + stage rebalancing.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — see [dualpipe](../systems/dualpipe.md) for bidirectional PP.
- Code: NVIDIA Megatron-LM — https://github.com/NVIDIA/Megatron-LM.
