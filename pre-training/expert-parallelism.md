# Expert Parallelism (EP)

*Depth — place different MoE experts on different GPUs; route tokens to experts via all-to-all.*

**TL;DR:** A MoE layer has K experts (typically 64–256). Placing all experts on one GPU defeats the whole point; replicating them defeats memory. **Expert parallelism** puts one (or a few) experts per GPU. Tokens get dispatched to their chosen experts via an **all-to-all** collective, processed locally, and **combined** back via another all-to-all. Two all-to-alls per MoE layer, across an "EP group" that usually spans multiple servers. GShard (2020) introduced the pattern; DeepSeek-V3 and Llama 4 operate at 256+ experts per layer.

**Prereqs:** [_parallelism](_parallelism.md), [_communication-primitives](../systems/_communication-primitives.md), [_moe](../architectures/_moe.md)
**Related:** [deepseek-moe](../architectures/deepseek-moe.md) · [tensor-parallelism](tensor-parallelism.md) · [dualpipe](../systems/dualpipe.md)

---

## What it is

In a Mixture-of-Experts layer, each token is routed to K-of-N experts (K is usually 1–2). Naïvely:
- Placing all N experts on one GPU: each token triggers N experts' weights to live on that GPU.
- Replicating experts on every GPU (like DDP): wastes memory proportional to N.

Expert parallelism places **experts on different GPUs**. For N experts across N_ep ranks, each rank holds `N/N_ep` experts. A token that needs expert `e` must be sent to the rank that owns expert `e`, processed there, and its output sent back.

EP is essentially **all-to-all data movement conditioned on routing decisions**.

---

## How it works

### The MoE layer forward

Input: `x ∈ ℝ^(B·S × H)` — a batch of tokens.

1. **Router** (runs locally on every rank): compute routing scores for each token. `scores = softmax(x · W_router)`. Each token picks top-K experts.
2. **Token dispatch (all-to-all)**: every rank sends each of its tokens to the rank that owns its chosen expert.
3. **Expert compute (local)**: each rank processes the tokens it received through its local experts.
4. **Token combine (all-to-all)**: results are sent back to the tokens' original rank.
5. **Weighted sum**: each token combines its top-K expert outputs weighted by router scores.

### The dispatch step — all-to-all

```python
# Pseudocode for MoE-EP forward (one rank, one layer)

def moe_layer_ep_forward(x_local, router, experts_local, ep_group):
    # x_local: [B * S_local, H] — tokens on this rank
    # experts_local: the subset of experts owned by this rank

    # Step 1: routing (local to each rank)
    scores = softmax(x_local @ router.weight)  # [tokens_local, num_experts_total]
    topk_experts, topk_weights = scores.topk(K)  # which experts, with what weights

    # Step 2: prepare dispatch
    # For each token, figure out which rank owns its chosen experts
    target_rank_per_token = topk_experts // experts_per_rank  # [tokens_local, K]

    # Bucket tokens by target rank
    buckets = {rank: [] for rank in range(ep_world_size)}
    for tok_idx, experts in enumerate(topk_experts):
        for e in experts:
            target_rank = e // experts_per_rank
            buckets[target_rank].append((tok_idx, e, x_local[tok_idx]))

    # Step 3: ALL-TO-ALL — send each rank's buckets[j] to rank j
    send_buffers = [pack(buckets[j]) for j in range(ep_world_size)]
    recv_buffers = all_to_all(send_buffers, ep_group)

    # Step 4: LOCAL expert compute
    # recv_buffers[j] contains tokens from rank j that need this rank's experts
    expert_outputs = {}
    for source_rank_tokens in recv_buffers:
        for (tok_idx, expert_id, token) in source_rank_tokens:
            local_expert_id = expert_id - ep_rank * experts_per_rank
            expert_outputs[(source_rank, tok_idx, expert_id)] = experts_local[local_expert_id](token)

    # Step 5: ALL-TO-ALL back — send expert outputs back to original ranks
    return_buffers = reorganize_outputs_by_source_rank(expert_outputs)
    received_outputs = all_to_all(return_buffers, ep_group)

    # Step 6: weighted combine
    combined = zeros_like(x_local)
    for tok_idx in range(tokens_local):
        for k in range(K):
            weight = topk_weights[tok_idx, k]
            combined[tok_idx] += weight * received_outputs[tok_idx, k]

    return combined
```

### The two all-to-alls

The critical primitive: **all-to-all** is a collective where **every rank sends a distinct payload to every other rank and receives distinct payloads from every other rank**.

- With N_ep ranks, all-to-all moves `N_ep · (N_ep − 1)` messages.
- Per-rank bandwidth: `O(M · N_ep)` bytes if each rank's total send volume is `M`.

For MoE: `M = B · S_local · H / N_ep_avg` per rank per all-to-all (where `N_ep_avg` reflects how evenly tokens distribute across experts).

See [_communication-primitives](../systems/_communication-primitives.md) for the collective-algorithm details.

### The backward pass

Symmetric:
1. Combine-all-to-all's gradient flows through a send/recv matching the forward's combine.
2. Expert gradient computed locally.
3. Dispatch-all-to-all's gradient is the reverse of the forward dispatch.

Bookkeeping is nontrivial — you must track exactly which tokens went where to reconstruct gradients correctly.

---

## The load-balance problem

EP's Achilles heel: **token distribution across experts is not uniform**.

If the router sends 50% of tokens to one hot expert, the rank holding that expert spends 50% of time on expert compute while other ranks idle. With 256 experts and millions of tokens, this imbalance adds up.

### Capacity factor (early MoE)

GShard / Switch Transformer introduced a **capacity factor c** (typically 1.0–1.25): each expert has a hard cap of `c · tokens_per_expert_avg` tokens per batch. Tokens above the cap are **dropped** (skip the MoE, go straight to the residual) or **re-routed** to the next-best expert.

- Low c: tight load balance but high drop rate → poor accuracy.
- High c: less imbalance but wasted capacity.

Modern MoE (DeepSeek-MoE, MoE-Plus) typically avoids hard drops and uses auxiliary losses or aux-loss-free load balancing instead.

### Auxiliary load-balancing loss (Switch Transformer)

Add a loss term `aux = α · N · Σ_e f_e · P_e` where `f_e` is fraction of tokens routed to expert e and `P_e` is average routing probability for expert e. Penalizes imbalance. α ≈ 0.01.

### Aux-loss-free (DeepSeek-V3)

DeepSeek-V3 (see [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md)) adds a **bias term** to each expert's routing score, adjusted online to equalize expert utilization. No auxiliary loss, cleaner gradient signal.

### Sequence-wise balance loss

Applies the balance loss within each sequence (not just within the global batch). Prevents one sequence from monopolizing a single expert. See [sequence-wise-balance-loss](../architectures/sequence-wise-balance-loss.md).

---

## Composing EP with other parallelism

EP has its own parallel group. A full MoE training run composes:

- **TP** (tensor parallel) inside attention + MLP.
- **EP** (expert parallel) inside MoE experts.
- **PP** (pipeline parallel) across layers.
- **DP / FSDP** outermost.

DeepSeek-V3 uses `TP × EP × PP × DP`, where:
- TP = 8 (within server, NVLink).
- EP = 8 or 16 depending on expert count.
- PP = 16.
- DP × FSDP fills the rest.

The EP group often overlaps with TP group for small EP or runs across a dedicated set of servers for large EP.

### EP's communication pattern

The **all-to-all** pattern in EP is expensive. Per token per layer: 2 all-to-alls × `H` bytes = `2 · H` bytes per token per layer per step.

For DeepSeek-V3 (H = 7168, 61 MoE layers): ~900 KB of all-to-all traffic per token per step. Over a batch of 16M tokens: **~14 TB of all-to-all per step**. Cross-server RoCE can be a bottleneck.

**Mitigations**:
- **Custom NCCL kernels**: DeepSeek-V3 uses highly-tuned MoE all-to-all kernels that overlap comm with compute.
- **Topology awareness**: keep EP within a node when possible (EP ≤ 8) or at most across two nodes.
- **Expert replication** (DeepSeek-V3 in later rounds): replicate "hot" experts across multiple ranks to spread load.

---

## Why it matters

- **Memory scales with expert count without replication.** N experts at one-per-rank = N× less per-rank memory than replication.
- **Enables frontier MoE.** DeepSeek-V3 (671B-total / 37B-active), Mixtral (8 experts), Qwen3-MoE, GLM-4.5-MoE — all use EP.
- **Distinct comm pattern from dense.** Dense training uses all-reduce / all-gather / reduce-scatter. MoE adds all-to-all — a fundamentally different collective with its own tuning space.
- **Composes cleanly with other axes.** EP is orthogonal to TP/PP/DP. Production MoE runs use all four.

---

## Gotchas & tricks

- **All-to-all dominates.** In many MoE runs, all-to-all is the largest single comm cost — larger than TP's all-reduces. Tune kernel and topology aggressively.
- **Load imbalance kills throughput.** A hot expert bottlenecks the entire EP group. Monitor per-expert utilization; react with rebalancing, aux loss tuning, or expert replication.
- **Capacity factor matters for older MoE codebases.** Newer codebases (DeepSeek) avoid hard caps. But if you're using GShard-style MoE, c is a critical knob.
- **Dropless MoE.** Some recent MoE variants (MegaBlocks, DeepSeek-MoE) eliminate the drop/re-route behavior entirely by using block-sparse GEMMs that handle variable expert loads natively.
- **EP group size.** EP = 8 is comfortable within a server. EP > 8 crosses into inter-server all-to-all, expensive. EP = 256 (Llama 4, DeepSeek-V3 large) requires careful topology.
- **Token permutation overhead.** Preparing send buffers (grouping tokens by target rank) has nontrivial CPU/GPU overhead for large batches. Fused "permute + all-to-all" kernels (e.g., from MegaBlocks) reduce this.
- **EP + TP on experts.** Large experts may also be TP-sharded internally (each expert split across multiple ranks). Nested parallelism — EP puts experts on ranks, TP splits each expert further.
- **Shared experts (DeepSeek).** DeepSeekMoE splits experts into "shared" and "routed." Shared experts are always active and replicated across ranks (not EP-sharded); routed experts are EP-sharded. Shared reduces the routing noise for common features.
- **Expert choice vs token choice.** Classical MoE: token picks top-K experts. Expert Choice Routing (Zhou 2022): each expert picks top-N tokens. Perfectly balanced by construction, but changes the training dynamics. Used in some Google MoE variants.
- **Numerical stability.** The all-to-all in BF16 with large N_ep can lose precision on summed gradients. FP32 reduce-scatter as compensation.
- **Overlapping all-to-all with attention.** DeepSeek-V3's DualPipe schedules attention of micro-batch k to overlap with MoE all-to-all of micro-batch k − 1. See [dualpipe](../systems/dualpipe.md).

---

## Sources

- Paper: *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* — Lepikhin et al., 2020, arXiv 2006.16668 — introduces expert parallelism + all-to-all dispatch.
- Paper: *Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* — Fedus et al., 2021, arXiv 2101.03961 — capacity factor, aux load-balance loss.
- Paper: *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale* — Rajbhandari et al., 2022, arXiv 2201.05596 — EP + TP + PP composition.
- Paper: *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* — Gale et al., 2022, arXiv 2211.15841 — dropless MoE via block-sparse GEMMs.
- Paper: *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* — Dai et al., 2024, arXiv 2401.06066 — fine-grained experts + shared experts.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — aux-loss-free balancing + custom all-to-all kernels + DualPipe-overlapped MoE.
- Paper: *Mixtral of Experts* — Mistral AI, 2024 — 8-expert MoE in production.
