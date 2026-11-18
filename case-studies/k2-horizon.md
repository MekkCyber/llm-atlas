# Case Study: K2-Horizon

*A six-model open-source fleet from MBZUAI's Institute of Foundation Models, spanning **0.9B dense** to **375B-A23B** mixture-of-experts. Released under the LLM360 definition of open — weights, intermediate checkpoints, pretraining data, training logs, and code. Two attention-block innovations (MoVA, softplus gate) and a diffusion-based self-speculator (Uno) sit on top of a ~22T-token pretraining corpus that includes ~10T synthetic tokens.*

**Related concepts:** [mova](../architectures/mova.md) · [softplus-attention-gate](../architectures/softplus-attention-gate.md) · [diffusion-decoder-adapter](../inference/diffusion-decoder-adapter.md) · [_moe](../architectures/_moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [mla](../architectures/mla.md) · [mtp](../pre-training/mtp.md) · [fp8-training](../pre-training/fp8-training.md) · [grpo](../post-training/grpo.md) · [rlvr](../post-training/rlvr.md) · [_speculative-decoding](../inference/_speculative-decoding.md)

---

## What this is

K2-Horizon is a *fleet* release — six models from MBZUAI's K2 team (Zhengzhong Liu, Hector Liu, Eric P. Xing, and ~90 co-authors), submitted to arXiv on 2026-11-16. The lead sizes advertised in the abstract are:

- **0.9B dense** — smallest model, same recipe family
- **36B-A4B MoE** — the paper's efficiency showcase; with MoVA + softplus gate it nearly matches a 32B dense model trained on the same data at **~1/7 the active parameters**
- **375B-A23B MoE** — the flagship; competitive with closed frontier systems on agentic and coding benchmarks

Three other sizes round out the fleet; four of the six models share one **~22T-token** pre-training corpus and recipe. The release follows the **LLM360** definition of open source: weights, data, intermediate checkpoints across training, logs, and the training stack (xLLM for pretraining, RL360 for post-training) are all published.

The paper matters because it (a) ships a frontier-tier MoE with full training transparency — rare at this scale — and (b) bundles three mechanistically novel pieces into one release: **MoVA** (MoE inside attention value projections), **softplus attention gating**, and the **Uno** diffusion decoder adapter for lossless self-speculative decoding.

---

## Architecture at a glance

The 375B-A23B flagship:

```
~375B total parameters / ~23B activated per token
MoE FFN                 — fine-grained expert pattern, top-k routing with
                          aux-loss-free balancing
Attention block         — MoVA (expert-routed value projections)
                          + softplus attention gate
                          + a KV-compression attention variant in the MLA family
RoPE positional encoding
MTP head (optional)     — for use with Uno at inference
```

The 36B-A4B model is the clean ablation case for the attention changes: it uses both MoVA and the softplus gate, keeps everything else matched to a 32B dense baseline, and approaches dense quality at **4B active** parameters.

The three in-house building blocks:

- **[Mixture-of-Value Attention (MoVA)](../architectures/mova.md).** Each head owns a pool of value projections and a token-dependent gate picks top-*k* per head per token. Queries and keys stay single-expert; only V is sparse-routed. This is MoE *inside* attention, not alongside the FFN.
- **[Softplus attention gate](../architectures/softplus-attention-gate.md).** Per-head gate using softplus — unbounded above, strictly positive, smooth — in place of a sigmoid gate, to avoid the "head cap" behavior sigmoid gates exhibit at long context.
- **[Uno — diffusion decoder adapter](../inference/diffusion-decoder-adapter.md).** A small diffusion head on top of the pretrained model that proposes several future tokens via denoising; speculative-sampling verify keeps acceptance lossless. Up to **2.2×** decode speedup, no separately-trained drafter.

---

## Training recipe

### Data

- **~22T-token** pretraining corpus shared across four of the six models.
- **~10T synthetic pretraining tokens** produced by K2's own synthesis pipelines — a very large synthetic fraction, in line with the direction of travel since 2025 (OLMo 2, Qwen 2.5, Llama 4) but much more explicit about it.
- Full data mixture, filters, decontamination pipelines, and deduplication steps are included in the release per the LLM360 definition.

### Pretraining stack

- **xLLM** — K2's pretraining framework, released with the paper.
- Training pipeline runs the standard four-stage shape (stability warmup → long constant LR → decay → stability tail), details in-paper. The abstract calls out no new learning-rate-schedule innovation.
- FP-mix precision details (FP8 vs BF16 split) and parallelism layout are published; the abstract does not spotlight them, so they follow the frontier default.

### Post-training

- **RL360** — K2's post-training / RL framework, also released.
- The abstract mentions reasoning and agentic specialization but does not detail an RL recipe. From the released framework and fleet composition, GRPO-style on-policy RL with verifiable rewards ([RLVR](../post-training/rlvr.md)) is the expected shape.
- The 36B-A4B and 375B-A23B models are targeted at coding and agentic benchmarks; the abstract frames them as "competitive with strong open models of similar size on agentic and coding benchmarks."

### Open-source commitments

K2-Horizon claims the **full LLM360 release**: intermediate checkpoints across training, weights, pretraining data, training logs, and training code (xLLM, RL360). In the current frontier-tier landscape this is the deepest open release since OLMo 2 and arguably deeper (the release spans five model scales, not one).

---

## Inference stack

The flagship ships with the **Uno** diffusion adapter wired into the inference path for self-speculative decoding. A standard speculative-sampling verify over the diffusion-drafted tokens keeps the output distribution identical to vanilla autoregressive decoding; K2 reports **up to 2.2× tokens-per-second** across the fleet.

This is the diffusion-as-drafter analogue of DeepSeek-V3's MTP-as-drafter idea ([mtp](../pre-training/mtp.md)): the drafting head lives in the main checkpoint, so there's no separately-trained draft model to maintain.

---

## Evaluation snapshot

The abstract reports competitive performance against strong open-source models of similar size on agentic and coding benchmarks and highlights the headline comparison: **36B-A4B (MoVA + softplus) ≈ 32B dense at ~1/7 the active parameters** on the same training data. Specific benchmark tables are in the paper; the abstract does not list numbers.

The most citation-worthy results from this release are likely to be:

1. The MoVA active-vs-dense parameter efficiency number.
2. The Uno lossless speedup across six model sizes.
3. The data-transparency delta — intermediate checkpoints for a 375B MoE, which no other frontier open-source release has shipped at this scale.

---

## Key takeaways

1. **MoE inside attention is a usable new axis.** MoVA is the first production-tier demonstration that expert routing on value projections — not just FFNs — yields active-parameter efficiency without breaking attention kernels.

2. **Attention gating shape matters at long context.** Softplus gates sidestep the bounded-gate saturation that sigmoid gates exhibit on long sequences. Cheap change, measurable quality win when paired with MoVA.

3. **Diffusion adapters replace trained-drafter speculative decoding.** Uno self-speculates with zero extra checkpoints and no training-time coupling to the target distribution. For teams shipping multiple sizes of one base model, this is a cleaner path than per-size EAGLE training.

4. **LLM360-style release is viable at 375B.** Intermediate checkpoints, full data, full code for a 375B-A23B MoE — no closed-model practical barrier is cited. Compare to DeepSeek-V3 (open weights only) or Llama (open weights and a data-mixture summary): K2-Horizon ships more.

5. **Synthetic pretraining data is now ~45% of the corpus.** 10T synthetic tokens inside a 22T-token recipe is the most explicit synthetic-fraction a frontier open release has published. The data-side release lets other teams audit whether this generalizes.

6. **A fleet, not a model.** Covering 0.9B → 375B in one release with a shared recipe trades flagship headlines for a broader research artifact — deliberate, and aligned with LLM360's mission of making the whole trajectory visible.

---

## What's still opaque

- **Full ablation of MoVA alone vs MoVA + softplus gate.** The abstract reports the combined effect; the paper presumably decomposes it, but the per-ingredient marginal contribution is the question for integrators.
- **The three unnamed sizes.** The abstract calls out 0.9B / 36B-A4B / 375B-A23B explicitly; the other three sizes in the six-model fleet aren't listed in the abstract text and need the paper or project page for detail.
- **Exact FP-mix precision schedule.** The release is framework-open but the abstract doesn't spotlight whether FP8 / MXFP8 training was used at the 375B scale.
- **Safety / refusal training.** Not characterized in the abstract. This is a frontier release and the full post-training evaluation of refusal / dangerous-capability behavior is not in the abstract summary.
- **Live comparison to DeepSeek-V4, Qwen 3, and closed frontier systems** — the paper's eval sweep presumably contains it; the abstract summarizes rather than ranks.

---

*Pairs well with:* the [DeepSeek-V3 case study](deepseek-v3.md) for architectural contrast — V3 is the KV-cache-savings reference (MLA), K2-Horizon is the attention-expert-routing reference (MoVA); and with [OLMo 2](olmo-2.md) for the open-trajectory comparison (same spirit, different scale).
