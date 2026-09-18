# Prerouted MoE — SSD-Streaming Inference via One-Layer-Ahead Routing Prediction
*Depth — closing the routing/read latency gap that blocks SSD-offloaded MoE inference.*

**TL;DR:** MoE inference from SSD-offloaded expert weights normally fails because layer $N{+}1$'s expert set can only be chosen *after* layer $N$'s output — so the reads for layer $N{+}1$ can't start early enough to hide behind compute. Edge0 (Lin et al. 2026) adds a **prerouter**: a lightweight per-layer head that predicts the *next* layer's routing decisions one token ahead. The prediction *is* the routing, so the staged expert set equals the routed set. An unmerged **recovery LoRA** trained on the student path repays quality lost to int4 quantization + routing substitution.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md)
**Related:** [../architectures/aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md) · [../quantization/fp8.md](../quantization/fp8.md) · [../quantization/_number-formats.md](../quantization/_number-formats.md)

---

## What it is

A streaming MoE inference engine for consumer hardware. The bottleneck it targets is **weight bytes**, not FLOPs: a 35B-class MoE at 4-bit is 19.5 GB, and sparsity shrinks the compute per token but not the bytes that must be resident to serve it.

Naive SSD offload buys nothing because the dependency chain
```
layer N compute → layer N output → layer N+1 routing → layer N+1 expert reads
```
gives the reads no head start. Prerouting breaks this chain by moving the routing decision *ahead* of the previous layer's output.

## How it works

- **Prerouter.** For each MoE layer $\ell$, train a small head that takes hidden states from layer $\ell - 1$ (or earlier) and predicts layer $\ell$'s expert selection *for the next token*. Its prediction is authoritative — the staged expert set from SSD is exactly the executed one. No dropped tokens, no re-fetches.
- **Recovery LoRA.** Int4 weight quantization plus routing substitution both degrade quality. A small LoRA is trained on the *student* path (i.e., with the prerouter's routing already in the loss) to reclaim the gap. Unmerged so it can be swapped or updated.
- **Overlap schedule.** SSD reads for layer $\ell$ are issued the moment layer $\ell - 1$'s prerouter output is available, giving the read a full layer's compute time to complete.

## Why it matters

- **Moves 35B-class MoEs onto 24 GB consumer GPUs.** On one 24 GB machine, Edge0 serves a 35B MoE at **20 tok/s** in **3 GiB peak active memory**, within a few points of the fp16 teacher across 5 public benchmarks.
- **An 8B tier runs on the same framework** — the pattern generalizes.
- **Open source.** Framework, checkpoints, adapters released — enables reproduction and downstream tinkering.

## Gotchas & tricks

- **Prerouter accuracy is the ceiling.** Mispredictions force fallback (re-fetch), which spikes latency. The paper measures top-$K$ agreement between prerouter and teacher router; report this alongside throughput.
- **LoRA is trained on the student path, not the teacher's.** Training the LoRA on teacher routing and then swapping in the prerouter reintroduces the substitution-quality gap.
- **SSD latency budget is fixed.** If SSD read latency exceeds the compute time of one MoE layer at your batch size, prerouting can't hide it — batch or context can't grow arbitrarily.
- **Not applicable when experts fit in HBM.** For 8B MoEs on a 24 GB card the offload story is moot; use standard batched decode.

## Sources

- Paper: *The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction* — Lin, Wang, Cai, Liu, Zeng, 2026, arXiv 2609.18063.
