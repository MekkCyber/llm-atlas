# BeaconKV
*Depth — training-free KV-cache compression that keeps entries revisited by future "thought revisits" in long CoT.*

**TL;DR:** In long reasoning traces some tokens re-attend to far-back plan context ("thought revisiting tokens", TRT). BeaconKV clusters query embeddings into a small set of **beacon queries** and scores each KV entry against the beacons, keeping the entries future revisits will want. Up to **5.8× memory reduction** and **>4.3× throughput** at near-full-cache accuracy, no training required.

**Prereqs:** [attention.md](../fundamentals/attention.md), [transformer-block.md](../architectures/transformer-block.md)
**Related:** [_kv-cache-compression.md](_kv-cache-compression.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A KV-cache eviction policy for long-reasoning inference (R1/QwQ-style models). Standard eviction assumes the recent-query attention pattern predicts what future decoding will re-attend to. In long CoT that assumption breaks: models periodically revisit early plan tokens, making "recent" a bad proxy.

## How it works

Offline / periodic:

1. Cluster the running query history in embedding space (a handful of clusters — the paper reports similarity groups small enough to represent by beacons per head).
2. Store one **beacon query** per cluster: a compact representative that stands in for that cluster's attention pattern.

At eviction time:

- For each KV entry, compute attention scores against **both** recent queries and beacon queries.
- Retain entries with high score under either set; evict the rest to hit the memory budget.
- Beacons are refreshed periodically as the trace grows.

No parameters are updated; the beacons live alongside the KV cache and grow with the number of distinct global attention patterns rather than sequence length.

## Why it matters

- **Long-CoT inference is KV-bound.** A 32K-token trace at 70B ≈ tens of GB of KV per request. Compressors that lose the plan tokens tank accuracy.
- **Training-free.** Drop-in for any pretrained model; no fine-tune or calibration step.
- **Data-shaped, not layer-shaped.** Where prior compressors set per-layer heuristics, BeaconKV's cluster count adapts to the trace's actual global-attention structure.
- **Composable with paged attention / continuous batching.** The beacon set is small and fixed-size; no impact on scheduler.

## Gotchas & tricks

- **Cluster count matters.** Too few beacons under-represent the revisit structure and evict plan tokens; too many defeat the compression. The paper's sweet spot is small (a few beacons per head).
- **Assumes clustering structure in queries.** Works well for chain-of-thought reasoning where revisits target a few plan regions; less-structured long-form generation may see smaller gains.
- **Combines with quantisation.** Beacon scoring on top of FP8 KV caches is a natural stack; the paper's speedups are before quantisation.
- **Beacon refresh cost.** Recomputing clusters is O(#queries), so schedule it at chunk boundaries rather than per-token.

## Sources

- Kim, Kim, Shim, Choi — *BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference* — 2026 — https://arxiv.org/abs/2609.04971
- Xiao et al., *StreamingLLM* (attention-sink baseline) — 2023 — https://arxiv.org/abs/2309.17453
- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference* — 2023 — https://arxiv.org/abs/2306.14048
