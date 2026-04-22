# DualPipe
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pipeline-parallelism scheduling algorithm that runs **two pipelines flowing in opposite directions** over the same set of GPUs and overlaps forward compute, backward compute, and MoE all-to-all communication so there's no idle time. Splits the backward pass into "backward-for-input" and "backward-for-weights" (à la ZeroBubble). Reduces pipeline bubbles to near zero at the cost of holding two model copies per GPU. Introduced in DeepSeek-V3.

**Prereqs:** basic understanding of pipeline parallelism (GPipe / 1F1B).
**Related:** none yet.

---

## What it is

Pipeline parallelism splits a model across devices along the layer axis: device 0 holds the first N/P layers, device 1 holds the next N/P, and so on for `P` pipeline stages. Micro-batches flow through forward in order, then backward in reverse. The problem is the **bubble**: at the start and end of each pipeline pass, GPUs further up or down the pipeline have nothing to do.

Classical schedules and their bubble ratios:

| Schedule | Bubble ratio | Memory |
|---|---|---|
| GPipe (all forward, then all backward) | `(P-1) / M` | high (stores all activations) |
| 1F1B (interleaved forward/backward) | `(P-1) / M` | moderate (one activation per stage) |
| Interleaved 1F1B | `(P-1) / (M·v)` where v = chunks per stage | more comm, same memory |
| ZeroBubble (1F1B variant) | ~0 | splits backward into B_input, B_weight |
| **DualPipe** | ~0, and comm-overlapped | 2× parameters (two copies) |

DeepSeek-V3 trained with `P = 16` pipeline stages. Under 1F1B that's a ~15-micro-batch bubble at each end; under DualPipe it's near zero, and cross-node MoE all-to-all traffic happens concurrently with compute.

---

## How it works

### The bidirectional schedule

Run **two pipelines** across the same GPUs simultaneously:
- Pipeline **F**: forward flows `GPU 0 → 1 → ... → P-1`, backward `P-1 → ... → 0`.
- Pipeline **B**: forward flows `GPU P-1 → ... → 0`, backward `0 → ... → P-1`.

Each GPU holds parameters for both pipelines' stages — so each GPU stores **two model copies**. Memory doubles for parameters, but the bubble nearly vanishes.

At any instant, a GPU in the middle of the pipeline is doing:
- Forward compute for pipeline F's micro-batch `k`
- Backward compute for pipeline B's micro-batch `k'`
- All-to-all dispatch / combine for whichever is active this tick

Because compute and communication are scheduled into different hardware paths (compute SMs vs IB / NVLink channels), they overlap — the GPU is never waiting on any one of them.

### Fine-grained stage splits

Each pipeline stage is further split into four compute phases per micro-batch:

```
stage forward = [ attention ] → [ all-to-all dispatch ] → [ MLP (MoE) ] → [ all-to-all combine ]
```

Backward is similarly split, and further divided into:
- **B_input**: gradient w.r.t. the stage's input activation.
- **B_weight**: gradient w.r.t. the stage's weights.

Splitting backward into `B_input` and `B_weight` is the ZeroBubble idea. `B_input` must run before the previous stage's backward can start (it's on the critical path), but `B_weight` can be deferred and filled into bubbles opportunistically. DualPipe uses this to pack compute into every cycle.

### Communication-compute overlap

Cross-node MoE all-to-all is expensive — potentially tens of milliseconds per stage. DualPipe schedules it so that while one pipeline's all-to-all is in flight, the other pipeline's compute is running on the same GPUs.

DeepSeek-V3's implementation reserves **20 of the 132 SMs** on each H800 GPU for communication (not dynamic — fixed allocation with warp specialization). The 112 remaining SMs handle compute. With this split:
- 20 SMs fully saturate the combined IB + NVLink bandwidth.
- 112 SMs handle all the matmul, attention, and FFN work.
- The two don't contend for scheduling resources.

### Communication path pipelining

Cross-node tokens traverse IB (between nodes) and NVLink (within the destination node). DualPipe's comm kernels pipeline both legs: while a token is being sent over IB for the next hop, the previous token's NVLink transfer is still completing. No intermediate host memcpy. This is specific to H800's reduced NVLink bandwidth (400 GB/s vs H100's 900 GB/s) — on full-bandwidth H100 the gains are smaller.

### Why it needs two copies of parameters

Each GPU is running forward for one pipeline and backward for the other *at the same time*. That requires two full copies of the stage's parameters — one frozen at the forward-in-progress version, one being updated by the backward pass. In practice:
- Forward uses the "current step" parameters.
- Backward accumulates gradients for the "previous step" parameters, which become "current" at the step boundary.

Memory doubles for parameters, but activations memory doesn't (same number of in-flight forward passes as 1F1B). For MoE models where parameters are already sharded across expert-parallel groups, the doubling is less painful than it sounds.

---

## Why it matters

- **Near-zero pipeline bubbles** at realistic model sizes. A 16-stage pipeline under 1F1B with 32 micro-batches has ~47% bubble efficiency loss; DualPipe at the same settings is effectively bubble-free.
- **Enables FP8 training to actually translate into wall-clock speedup.** FP8 gives 2× matmul throughput, but if the pipeline is bubble-heavy, utilization drops below the bubble floor and the 2× is wasted. DualPipe keeps utilization high enough that FP8's gains land in measured wall-clock.
- **Purpose-built for cross-node MoE.** MoE training's all-to-all is the main bandwidth cost. DualPipe overlapping it with compute means the expensive communication doesn't show up as serial time.
- **Makes H800 training practical at frontier scale.** H800's reduced interconnect means a naive schedule would be bandwidth-bound. DualPipe plus the 20-SM comm kernel plus node-limited routing together keep H800s close to H100 efficiency for DeepSeek-V3's workload.

---

## Gotchas & tricks

- **Parameters doubled.** Budget for 2× parameter memory per GPU. For a 671B MoE sharded across expert parallelism, per-GPU parameters are ~5B active — doubled to ~10B, still fits on H800 (80GB). On smaller GPUs this might not be feasible.
- **The 20-SM comm reservation is empirical.** More SMs for comm saturates the IB/NVLink link; fewer underuses it. 20 is what hit the sweet spot on H800. On H100 the number will be different (fewer, since more bandwidth per SM-minute).
- **Warp specialization is required.** Normal CUDA code lets the block scheduler assign warps freely; DualPipe's comm kernels explicitly pin specific warps to communication roles (IB send, IB receive, NVLink send, NVLink receive). Without this, the scheduler mixes comm and compute warps, and you lose the non-contention property.
- **Backward split requires careful gradient bookkeeping.** `B_input` and `B_weight` must use consistent activation/parameter snapshots. Bugs here produce gradients that are off by one step — subtle, slow to diagnose.
- **Not a free win for small models.** On a 2-stage pipeline with few micro-batches, bubbles are small anyway; DualPipe's overhead (2× params, scheduling complexity) isn't justified. It's a 8+ stage / long-run / MoE-heavy win.
- **Doesn't replace expert parallelism.** Pipeline is along layers; expert parallelism is across experts within a layer. DualPipe operates in the pipeline dimension only. MoE all-to-all still runs across expert-parallel groups; DualPipe just overlaps it with compute.
- **Interaction with tensor parallelism.** DeepSeek-V3 explicitly uses **no tensor parallelism in training** — DualPipe + expert parallelism + ZeRO-1 is enough. If you add TP, the TP all-reduces don't obviously overlap with DualPipe's comm schedule, and you may need to rework the warp specialization.
- **Implementation is open-source but non-trivial.** DeepSeek released the comm kernels (DeepEP) but the full DualPipe scheduler is complex. Expect to spend real engineering time reproducing on a different cluster topology.

---

## Sources

- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — introduces DualPipe in §3.2.1–3.2.2.
- Paper: *Zero Bubble Pipeline Parallelism* — Qi et al., 2023 — the ZeroBubble B_input / B_weight split DualPipe builds on.
- Paper: *GPipe* — Huang et al., 2018 — the original pipeline-parallel baseline.
- Paper: *Megatron-LM 1F1B* — Narayanan et al., 2019 — the 1F1B schedule DualPipe improves on.
- Repo: *DeepEP* — DeepSeek's open-sourced MoE communication kernels; the comm layer underneath DualPipe.
