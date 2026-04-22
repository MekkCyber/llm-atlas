# Case Study: OLMo 2

*AI2's fully open 7B / 13B / 32B base and instruct models — not because the final scores are frontier, but because they ship the recipe, data, code, and every intermediate checkpoint. The best reproducibility reference in the LLM literature.*

**Related concepts:** [qk-norm](../architectures/qk-norm.md) · [reordered-norm](../architectures/reordered-norm.md) · [z-loss](../fundamentals/z-loss.md) · [mid-training](../pre-training/mid-training.md) · [model-souping](../pre-training/model-souping.md) · [rlvr](../post-training/rlvr.md)

---

## What this is

OLMo 2 is Allen AI's follow-up to OLMo 1, released late 2024 (7B, 13B) and early 2025 (32B). It's a decoder-only transformer trained from scratch, hitting parity with Llama 3.1 and Qwen 2.5 at the same parameter counts — using only open data, open code, and a fully public training trajectory.

The interesting contribution is not the final weights. It's:

1. A concrete, ablated **stability recipe** (QK-norm + reordered-norm + z-loss + small init) that closes the loss-spike failure modes of 2023-era large runs.
2. A **two-stage training curriculum** where a small curated Stage-2 mix ("Dolmino") disproportionately shapes benchmark performance.
3. **Full openness**: every intermediate checkpoint, every data artifact, every eval. Other papers describe recipes; OLMo 2 lets you re-run them.

Read this report if you want to know *how* a model at this scale is trained, not just *that* it was.

---

## Architecture — stability-first transformer

```
Input tokens
    │
    ▼
[Embed + RoPE]
    │
    ▼
  ┌─────────── Block (×L) ─────────────┐
  │                                     │
  │   x ──┬─────────────────────────────┤
  │       │                             │
  │       ▼                             │
  │   [RMSNorm-free sub-layer input]    │
  │   [Attention w/ QK-norm]            │
  │       │                             │
  │       ▼                             │
  │   [RMSNorm on output]   ← reordered │
  │       │                             │
  │       ▼                             │
  │   x + out  ───┬─────────────────────┤
  │               │                     │
  │               ▼                     │
  │         [SwiGLU FFN]                │
  │               │                     │
  │               ▼                     │
  │         [RMSNorm on output]         │
  │               │                     │
  │               ▼                     │
  │         x₁ + out ───────────────────┤
  └─────────────────────────────────────┘
    │
    ▼
[Final RMSNorm]
    │
    ▼
[LM head → logits → CE + z-loss]
```

The architectural changes from OLMo 1, each independently ablated:

- **[Reordered norm](../architectures/reordered-norm.md)** — RMSNorm on the *output* of each sub-layer before adding to the residual. Neither pre-norm nor classic post-norm; their variant keeps the residual stream clean and sub-layer contributions bounded.
- **[QK-norm](../architectures/qk-norm.md)** — per-head RMSNorm on Q and K before the attention dot product. Prevents attention-logit drift.
- **[Z-loss](../fundamentals/z-loss.md)** — `α · (log Z)²` added to CE. Prevents output-logit drift, which was the other spike class.
- **Small-std initialization** and careful residual-branch scaling.

Everything else is standard (RoPE, SwiGLU, no biases). The contribution is the *stability package*, not the individual pieces.

---

## Training — two stages, annealed schedule

### Stage 1: breadth

| | 7B | 13B |
|---|---|---|
| Tokens | ~4T | ~5T |
| Data | OLMo-Mix-1124 | OLMo-Mix-1124 |
| LR schedule | Warmup → cosine to plateau | Warmup → cosine to plateau |

**OLMo-Mix-1124** (see [dolma.md](../data/dolma.md)) is the new pretraining data mix — a successor to Dolma 1.7. Common Crawl-dominant with StarCoder, arXiv, pes2o, Wikipedia, and Stack Exchange, all [deduped](../data/deduplication.md) (URL + paragraph + n-gram), PII-scrubbed, [quality-filtered](../data/quality-filtering.md) with a FastText classifier, and [decontaminated](../data/decontamination.md) against eval benchmarks. Every document is attributable to its source.

Stage 1 gets you a base "world model" — broad knowledge, unremarkable benchmarks.

### Stage 2: [mid-training](../pre-training/mid-training.md)

| | 7B | 13B |
|---|---|---|
| Tokens | ~50B | ~100B |
| Data | **Dolmino Mix 1124** | Dolmino Mix 1124 |
| LR schedule | **Anneal plateau → ~0** | Anneal plateau → ~0 |

**Dolmino Mix 1124** (see [dolma.md](../data/dolma.md)) is the curated late-stage mix: [heavily quality-filtered](../data/quality-filtering.md) web, math (OpenWebMath, GSM-style), code (high-rated GitHub, Stack Exchange), academic (arXiv, pes2o), FLAN-style instruction text, and synthetic rewrites. Small compared to Stage 1 but high quality.

This is where the benchmark numbers are made. Ablations in §5 of the report show: swap Dolmino for more Stage-1 data and you lose 5–10 points on GSM8K / MATH / HumanEval. Swap it in and you gain them back. A ~1% tail of training dominates the downstream picture.

### Optimizer & infra

- **AdamW**, bf16 mixed precision (not FP8 — contrast DeepSeek-V3).
- **Gradient clipping tightened** relative to OLMo 1; weight decay applied to all parameters including embeddings.
- **FSDP** for 7B/13B; 32B brings in additional parallelism.
- H100 clusters; 7B is in the low-thousands of GPU-days.

---

## Model souping

The final release weights are a [**uniform soup**](../pre-training/model-souping.md) of multiple Stage-2 runs that share the Stage-1 parent but use different data orderings and seeds. Averaging the state dicts element-wise produces the shipped checkpoint.

Why it works: the Stage-2 runs share a parent, so they're in the same loss basin (linear mode connectivity holds). Averaging concentrates mass near the basin floor, picking up a small but real benchmark lift over any single run — essentially free given the candidate runs already existed.

---

## Post-training — the Tülu 3 recipe

OLMo 2 Instruct is produced by applying the **Tülu 3** post-training pipeline to the base model:

```
Base  ──▶  SFT (curated instructions)
       ──▶  DPO (preference-tuned style + refusal)
       ──▶  RLVR (verifiable reward signal on math, code, format)
```

See [rlvr.md](../post-training/rlvr.md) for the key final step: PPO/GRPO with programmatic verifiers (exact-match for math, test-pass for code) instead of a learned reward model. Cheaper, not reward-hackable, and the major reason Tülu 3 / OLMo 2 Instruct scores well on reasoning benchmarks relative to its SFT+DPO-only peers.

---

## Results snapshot

| Model | MMLU | GSM8K | MATH | HumanEval | Average |
|---|---|---|---|---|---|
| OLMo 2 7B | ~62 | ~68 | ~33 | ~65 | on par with Llama 3.1 8B |
| OLMo 2 13B | ~68 | ~78 | ~41 | ~71 | on par with Qwen 2.5 14B |
| OLMo 2 32B | ~74 | ~85 | ~52 | ~78 | GPT-3.5 / Qwen 2.5 32B territory |

Numbers approximate; check the report tables for the exact eval harness versions and prompt formats.

The point is parity with closed-data peers at the same scale, using only released data. Previously no fully-open model at this scale had reached that bar.

---

## What's actually released

| Artifact | OLMo 2 has it |
|---|---|
| Final weights (base + instruct) | ✓ |
| Pre-training data mix | ✓ (OLMo-Mix-1124, Dolmino Mix 1124) |
| Post-training data | ✓ (Tülu 3 mixes) |
| Training code | ✓ (`OLMo-core` repo) |
| **Every intermediate checkpoint** | ✓ |
| Eval harness | ✓ |
| Training logs / wandb runs | ✓ |

This is what distinguishes OLMo 2 in the current literature. You can load a Stage-1 checkpoint at step 100k and resume, or diff checkpoint k vs. k+1 to study optimization dynamics, or retrain Stage 2 with a different mix. Nothing else at this scale supports that.

---

## Key takeaways

1. **Stability is a design choice, not luck.** [QK-norm](../architectures/qk-norm.md) + [reordered-norm](../architectures/reordered-norm.md) + [z-loss](../fundamentals/z-loss.md) + careful init together close the loss-spike failure mode. Apply to any ground-up training run.

2. **The late-stage mix dominates the benchmark picture.** The last 1–2% of tokens shifts scores more than any single change to the first 98%. Spend engineering effort on Stage 2's data, not just Stage 1's.

3. **[RLVR](../post-training/rlvr.md) > RLHF when the domain allows it.** Ground-truth verifiers are cheaper, robust to hacking, and scalable to synthetic curricula. SFT + DPO cover the non-verifiable axes; RLVR closes the verifiable gap.

4. **[Model souping](../pre-training/model-souping.md) is free quality.** Uniform-average several Stage-2 runs that share a parent. A rounding error of cost, a small but consistent lift.

5. **Openness is tractable at this scale.** The work involved in releasing Dolmino, OLMo-core, and the checkpoint stream is engineering effort, not compute — a small lab could have done it. The reason only AI2 did is organizational, not technical.

---

*Pairs well with:* the [Llama 3 Herd](https://arxiv.org/abs/2407.21783) report for the same-era recipe at 70B+ scale, and [Tülu 3](https://arxiv.org/abs/2411.15124) for the post-training side in full detail.
