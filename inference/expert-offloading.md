# MoE Expert Offloading
*Depth — serving a Mixture-of-Experts model on a single GPU by keeping experts in host memory and paging them in on demand.*

**TL;DR:** A MoE model has far more total parameters than any one token activates, so most experts are idle at any given step. Expert offloading keeps the full expert bank in CPU/host memory and copies only the per-token activated experts over PCIe into GPU memory when the router selects them. Done naïvely, PCIe transfer dominates per-token latency. Modern systems like **Specter** mitigate this with *speculative prefetch*: a cheap quantized draft model runs several tokens ahead and uses its own router decisions to prefetch the right experts before the main model needs them, hiding most of the transfer cost.

**Prereqs:** [_moe](../architectures/_moe.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [fp8](../quantization/fp8.md), [_number-formats](../quantization/_number-formats.md), [deepseek-moe](../architectures/deepseek-moe.md)

---

## What it is

A serving-system technique for large MoE models (DeepSeek-V3 scale and up) that don't fit in a single GPU's HBM. Instead of multi-GPU tensor-/expert-parallelism, the full weight set lives in host memory and the shared backbone + a working subset of experts live in HBM; experts are pulled in as the router activates them.

## How it works

Per decode step:

1. The router sees the current hidden state and selects top-$k$ experts for the next token (typical $k \in \{1, 2, 8\}$).
2. Each selected expert's weights must be present in HBM for the layer's forward to proceed. Missing experts are copied from host via PCIe/NVLink.
3. The expert pool in HBM is managed with an **LRU-style cache** — recently-used experts stay resident; cold ones are evicted.

The dominant cost is step 2's PCIe transfer. A single expert can be hundreds of MB of weights; even at PCIe Gen5 bandwidth, serial transfers per token blow the latency budget.

**Specter's addition** (2025): a quantized draft model decodes several tokens ahead on the GPU. Each draft step produces its own *router outputs* — a probability over experts. Those routing probabilities are used as **prefetch hints**: start copying the predicted experts over PCIe *before* the main model's router asks for them. Speculative decoding verifies the draft tokens in parallel; the expert prefetch mostly overlaps with compute even when the draft tokens are rejected, because routing decisions are usually correct even when final token predictions differ.

## Why it matters

- **Enables single-GPU serving of frontier MoE models.** DeepSeek-V3's 671B total / 37B active doesn't fit in 80 GB HBM, but 37B active does; offloading unlocks the architecture at consumer-GPU scale.
- **Decouples total parameters from HBM.** Lets a lab iterate on bigger MoEs than its serving hardware supports natively.
- **Attacks the critical path.** Weight quantization lowers *bandwidth demand*; expert offloading + prefetch lowers *latency on the critical path* — a complementary axis.

## Gotchas & tricks

- **Router accuracy of the draft model is what matters**, not its token accuracy. A draft that picks the right experts with high recall is enough to drive prefetch, even if its token predictions are often rejected.
- **PCIe contention with other I/O** (model loading, data pipelines) can starve the prefetch bandwidth; dedicate a PCIe root complex to the expert channel if possible.
- **Cold-start penalty.** First tokens after a prompt switch pay the full miss cost until the LRU cache warms.
- **Shared/root experts should be resident.** DeepSeekMoE-style always-on experts belong in HBM permanently; only routed-expert slots need offloading logic.
- **Quantize the draft, not the experts you're offloading.** Quantizing offloaded experts saves PCIe bandwidth but costs quality; quantizing the draft saves GPU compute without touching the main model.
- **Does not help compute-bound regimes** (large batch, long prefill); the win is at memory-bound decoding with batch size 1–few.

## Sources

- Paper: *Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading* — Wang et al., 2025 — [arXiv:2511.14102](https://arxiv.org/abs/2511.14102).
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — the model class this technique primarily targets.
- Related: *Mixtral of Experts* — Mistral AI, 2024 — earlier open MoE where offloading was prototyped by the community.
