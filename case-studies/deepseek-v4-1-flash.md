# Case Study: DeepSeek-V4.1-Flash

*A 552B multimodal MoE tuned for long-horizon agentic workloads where prefill and KV-cache bandwidth are the bottleneck. Its four architectural moves — an asymmetric prefill/decode split, cross-layer KV reuse, FP4 KV caching, and bounded sliding-window replay — collectively cut the global KV footprint to ~890 bytes/token (~4× denser than V4-Flash) while beating it on quality. The interesting story is that every knob was co-designed for agent-shaped workloads: input-heavy, long-context, many turns.*

**Related concepts:** [csa2](../architectures/csa2.md) · [ced](../architectures/ced.md) · [../inference/fp4-kv-cache.md](../inference/fp4-kv-cache.md) · [../inference/swa-bounded-replay.md](../inference/swa-bounded-replay.md) · [mla](../architectures/mla.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [fp8](../quantization/fp8.md) · [_number-formats](../quantization/_number-formats.md) · [fp8-training](../pre-training/fp8-training.md) · [deepseek-v3 case study](deepseek-v3.md)

---

## What this is

DeepSeek-V4.1-Flash, released September 2026 by DeepSeek-AI. A multimodal Mixture-of-Experts decoder-oriented transformer with a **Causal Encoder-Decoder (CED)** front: 552B total backbone parameters, activating **16B per token during decode** but only **8B per token during prefill**, with support for contexts up to **one million tokens**. Trained on 45T multimodal tokens.

The paper's stated bottleneck: with long-horizon agents dominating workloads, prompts dwarf outputs. Prefill FLOPs and KV-cache HBM/SSD bandwidth are now the primary cost drivers, not decode FLOPs. V4.1-Flash is the first frontier release where every knob is set for that regime:

- **CED** halves prefill compute vs. decode.
- **CSA2** shares one compressed KV projection across layers (extending MLA's per-layer compression to a cross-layer regime).
- **FP4 KV caching** compresses the remaining cached values to 4 bits.
- **SWA Bounded Replay** bounds sliding-window replay so the *persistent* KV on SSD/host stays ~1/8 of V4-Flash's.

Result: 890 bytes per token in the global (HBM-resident) KV cache — roughly 1/4 of V4-Flash's per-token footprint — while quality goes *up*.

---

## Architecture at a glance

```
Causal Encoder-Decoder (CED)
  ├─ prefill tower:  ~8B activated params/token
  └─ decode  tower: ~16B activated params/token
     (asymmetric MoE routing; encoder is lighter,
      decoder carries the generation-side experts)

attention: Compressed Sparse Attention 2 (CSA2)
  cross-layer KV cache reuse
  + sparse-attention pattern
  + FP4 KV caching (E2M1 or block-scaled variant)

MoE: fine-grained + shared experts (DeepSeekMoE line)
     aux-loss-free bias-based balancing (unchanged from V3)

context window: 1 000 000 tokens (long-horizon agent workloads)
modality: text + vision, multimodal pretraining on 45T tokens

deployment
  ├─ HBM-resident KV cache:  ~890 bytes / token global
  ├─ SSD/host persistent KV: ~1/8 of V4-Flash
  └─ SWA Bounded Replay for sliding-window agent turns
```

All four boxed innovations get their own concept files in the **Related concepts** line above.

---

## The four architectural moves

### 1. Causal Encoder-Decoder (CED) — asymmetric prefill/decode

Traditional decoder-only transformers run the same weights during prefill (all-input, one forward pass) and decode (per-token). Agent workloads are input-heavy, so prefill dominates cost per request. CED splits the network so a smaller expert subset activates on prefill:

- **Prefill:** 8B activated parameters/token.
- **Decode:** 16B activated parameters/token.

The "encoder" is the lighter prefill-side subset; the "decoder" is the heavier generation-side. Both share the KV path via CSA2. Depth file: [ced](../architectures/ced.md).

### 2. Compressed Sparse Attention 2 (CSA2) — cross-layer KV reuse

MLA (in V2/V3) compressed each layer's KV cache into a low-rank latent. CSA2 goes further: one compressed KV latent is **shared across multiple layers**, combined with a sparse-attention pattern per layer that decides which tokens to read. Where MLA saves a factor per layer, CSA2 saves across layers. Depth file: [csa2](../architectures/csa2.md).

### 3. FP4 KV caching

Applied on top of CSA2's compression: the remaining cached values are stored at 4-bit precision (E2M1 or block-scaled variant, following the DeepSeek FP-precision line from V3's FP8). Combined with CSA2's cross-layer reuse, this is what gets the global KV cache down to ~890 B/token. Depth file: [fp4-kv-cache](../inference/fp4-kv-cache.md).

### 4. SWA Bounded Replay — SSD/host cache shaping

For agent workloads that repeatedly re-use large context, part of the KV cache is persisted to SSD or host memory (not HBM). Sliding-window attention naturally introduces "replay" — recomputing edges of prior windows. SWA Bounded Replay bounds this replay so the persistent cache stays compact: another ~2× smaller than the global HBM cache. Depth file: [swa-bounded-replay](../inference/swa-bounded-replay.md).

---

## Training recipe (from abstract)

- **Pretraining data:** 45T multimodal tokens (text + vision).
- **Backbone:** 552B total parameters, MoE with fine-grained + shared expert structure.
- **Post-training:** "comprehensive" per the abstract, covering text-based and multimodal agentic scenarios. Specifics not enumerated in the abstract; expect a mix of SFT on multimodal instruction data, RL with agentic environment rewards, and possibly distillation from a reasoning sibling.
- **Precision:** builds on the V3-line FP8 training; FP4 for KV cache is the deployment-side extension.

Training compute and cost were not disclosed in the abstract. Model checkpoints released at `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`.

---

## Deployment / inference regime

The paper's framing is explicit: agents make prefill and KV-cache the cost drivers, not decode FLOPs. Concrete deployment metrics from the abstract:

| Metric | DeepSeek-V4.1-Flash | DeepSeek-V4-Flash |
| --- | --- | --- |
| Global KV cache (HBM) | ~890 B/token | ~4× larger |
| Persistent KV cache (SSD/host) | (~890 B / 8) B/token | ~8× larger |
| Prefill-activated params/token | 8B | (larger, symmetric) |
| Decode-activated params/token | 16B | (same as prefill) |
| Context window | 1 000 000 tokens | shorter |
| Quality vs. V4-Flash | **higher** | baseline |

Absolute serving numbers (throughput, latency, cost/1k tokens) not in the abstract — expect a deployment appendix in the full paper.

---

## Key takeaways

1. **Prefill is the new bottleneck.** V4.1-Flash is the first frontier system that assumes so and re-partitions the compute graph accordingly. Asymmetric prefill/decode (CED) is the concrete answer; expect other MoE lines to follow.

2. **KV cache compression stacks.** MLA → CSA2 → FP4 → SWA Bounded Replay: each layer of compression multiplies against the others. Total effect at deployment: 4× smaller HBM, 8× smaller SSD. Ambitious stack of small tricks rather than one big idea.

3. **FP4 is the new FP8.** DeepSeek's precision-frontier line continues: FP8 training in V3, FP4 KV caching in V4.1-Flash. Serving-side low precision is now decoupled from training-side low precision; both matter.

4. **Long-horizon agent workloads have their own architecture.** Not just a bigger context window — the whole prefill/decode/HBM/SSD stack is different from the chatbot regime. This is the first paper that architects for that difference.

5. **Weights released.** Continues the V3/R1 tradition. Community can reproduce the deployment stack.

---

## What's still opaque

- **CSA2 mechanism details.** How exactly cross-layer KV reuse is done (weight sharing, learned mapping, layer grouping) is under-specified in the abstract; full paper needed.
- **FP4 recipe.** Whether it's block-scaled E2M1, MXFP4, or a custom variant. Given the DeepSeek precision line, likely a custom variant with per-tile scaling similar to their FP8 recipe.
- **Training cost.** Not disclosed in the abstract. Compare to V3's $5.576M — 552B multimodal + 45T tokens is presumably significantly higher.
- **Post-training pipeline.** SFT/RL details, reasoning distillation source (a V4-R? a V3.5-R1?) — none enumerated.
- **CED implementation.** Whether the prefill/decode split is a hard partition (different experts) or a soft one (different activation gate). Full paper needed.
- **Multimodal training details.** 45T multimodal token composition (text/image/video ratio, curation) not disclosed.

---

*Pairs well with:* [deepseek-v3](deepseek-v3.md) for the ancestor stack (MLA, DeepSeekMoE, FP8, DualPipe). V3 is the frontier-training reference; V4.1-Flash is the frontier-serving reference. Different halves of the same optimization surface.
