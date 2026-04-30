# Case Study: Llama 3

*Meta's July 2024 release: a family of dense Transformer models at 8B, 70B, and 405B parameters, trained on 15.6T tokens with 4D parallelism on 16K H100s. The canonical production-recipe paper for frontier-scale dense models. Open weights.*

**Related concepts:** [chinchilla-scaling](../pre-training/chinchilla-scaling.md) · [downstream-scaling-laws](../pre-training/downstream-scaling-laws.md) · [_parallelism](../pre-training/_parallelism.md) · [data-parallelism](../pre-training/data-parallelism.md) · [fsdp](../pre-training/fsdp.md) · [tensor-parallelism](../pre-training/tensor-parallelism.md) · [pipeline-parallelism](../pre-training/pipeline-parallelism.md) · [context-parallelism](../pre-training/context-parallelism.md) · [sequence-parallelism](../pre-training/sequence-parallelism.md) · [gqa](../architectures/gqa.md) · [intra-document-mask](../architectures/intra-document-mask.md) · [rope](../fundamentals/rope.md) · [dpo](../post-training/dpo.md) · [reward-modeling](../post-training/reward-modeling.md) · [capability-experts](../post-training/capability-experts.md) · [knowledge-probe-hallucination](../post-training/knowledge-probe-hallucination.md) · [rejection-sampling](../post-training/rejection-sampling.md) · [data-mix](../data/data-mix.md) · [annealing-as-data-eval](../pre-training/annealing-as-data-eval.md) · [_communication-primitives](../systems/_communication-primitives.md) · [llama-guard](../safety/llama-guard.md) · [prompt-guard](../safety/prompt-guard.md) · [code-shield](../safety/code-shield.md) · [rainbow-teaming](../safety/rainbow-teaming.md) · [uplift-evaluation](../safety/uplift-evaluation.md) · [vit](../multimodal/vision/vit.md) · [clip](../multimodal/vision/clip.md) · [cross-attention-adapter](../multimodal/vision/cross-attention-adapter.md) · [best-rq](../multimodal/audio/best-rq.md) · [conformer](../multimodal/audio/conformer.md) · [mmlu-pro](../evaluation/mmlu-pro.md) · [gpqa](../evaluation/gpqa.md) · [bfcl](../evaluation/bfcl.md) · [mgsm](../evaluation/mgsm.md)

---

## What this is

**Llama 3**, released July 2024 by Meta. arXiv 2407.21783 (v1 Jul 2024, v3 Nov 2024). Three dense Transformers: **8B, 70B, 405B**. All multilingual, 128K context, tool-use enabled (instruct). Pretraining: **~15T multilingual tokens** (~8× Llama 2); flagship 405B on **15.6T tokens** with **3.8 × 10²⁵ FLOPs** (~50× Llama 2 70B compute). Trained on **up to 16,000 H100 GPUs**.

The honest read: Llama 3 isn't one dominant algorithmic insight — it's a carefully-documented **production recipe** at frontier scale. The value is in the details: 4D parallelism, downstream-scaling-law methodology, DPO with auxiliary NLL loss, capability experts for SFT-data generation, compositional multimodal via cross-attention.

This case study is structured by technical contribution, not paper section order. Each block links to its own depth page.

---

## 0. The three levers (Sec. 1)

Llama 3's stated design philosophy — *"three key levers in the development of high-quality foundation models: data, scale, and managing complexity"* (l. 32–33):

- **Data** (~15T tokens, ~8× Llama 2) — both quantity and quality up.
- **Scale** (3.8 × 10²⁵ FLOPs, ~50× Llama 2 70B, ~compute-optimal).
- **Managing complexity**: *"we opt for a standard dense Transformer model architecture... rather than for a mixture-of-experts model to maximize training stability"* (l. 64–66). Post-training uses **DPO over PPO** for similar reasons: simpler, more stable at scale.

This framing explains most of the paper's decisions. The paper avoids novelty for novelty's sake and chooses "does this scale reliably?" at every step.

---

## 1. Architecture

Dense Transformer — the same recipe as Llama 2 with targeted changes.

| Spec | 8B | 70B | 405B |
|---|---|---|---|
| Layers | 32 | 80 | 126 |
| d_model | 4,096 | 8,192 | 16,384 |
| FFN dim | 14,336 | 28,672 | 53,248 |
| Q heads | 32 | 64 | 128 |
| KV heads ([gqa](../architectures/gqa.md)) | 8 | 8 | 8 |
| Head dim | 128 | 128 | 128 |
| Peak LR | 3 × 10⁻⁴ | 1.5 × 10⁻⁴ | 8 × 10⁻⁵ |
| Vocab | 128,000 | 128,000 | 128,000 |
| [RoPE](../fundamentals/rope.md) base θ | 500,000 | 500,000 | 500,000 |

Changes vs Llama 2 (Sec. 3.2):
- **[GQA](../architectures/gqa.md) with 8 KV heads** across all sizes. Shrinks KV cache 8× at inference.
- **[Intra-document attention mask](../architectures/intra-document-mask.md)**. Blocks attention across document boundaries in packed sequences. Small effect at short context; critical at 128K.
- **128K vocabulary** (up from 32K). Compression ratio 3.17 → 3.94 chars/token.
- **RoPE θ = 500,000** (up from 10,000). Pure ABF scaling for long context (no YaRN, no NTK-aware-by-parts).
- Same SwiGLU, RMSNorm, no biases.

Why dense, not MoE: *"to maximize training stability"* — one sentence. Operational simplicity at 405B scale.

---

## 2. Scaling laws — the downstream-accuracy extension

The paper's main methodological novelty. See [downstream-scaling-laws](../pre-training/downstream-scaling-laws.md) and [chinchilla-scaling](../pre-training/chinchilla-scaling.md).

Two-stage fit:
1. **compute → normalized NLL/char** (linear, from IsoFLOPs sweep at 40M–16B / 6 × 10¹⁸–10²² FLOPs).
2. **NLL → accuracy** (sigmoidal, anchored with Llama 2 models).

Chain to predict benchmark accuracy from compute. Extrapolates over 4 orders of magnitude; slightly underestimates 405B on ARC Challenge.

Compute-optimal fit (Eq. 10):

```
N*(C) = A · C^α   with (α, A) = (0.53, 0.29)
```

At C = 3.8 × 10²⁵: predicts **402B on 16.55T tokens** — basis for Llama 3's 405B/15.6T.

---

## 3. Pretraining data

The most detailed data section in any modern open tech report. See [data-mix](../data/data-mix.md) and related.

### The 4-layer web pipeline (Sec. 3.1.1)

1. **PII + safety filter** (domain-level blocklists).
2. **Text extraction + cleaning**: custom HTML parser, preserves math + code, keeps imagealt; removes markdown markers.
3. **Deduplication (3 levels)**: URL (keep-most-recent), document (**MinHash**), line (ccNet-style, remove lines appearing >6× in 30M-doc buckets).
4. **Heuristic filtering**: n-gram coverage, dirty-word, KL-divergence.

### Model-based quality filtering (Sec. 3.1.1)

- **fasttext** (Wikipedia-reference classifier, from Llama 1 lineage).
- **DistilRoBERTa** scorer distilled from Llama-2-chat quality labels.

Code + math: **DistilRoBERTa classifiers** targeting STEM-reasoning and code-interleaved-with-NL.
Multilingual: fasttext LID across **176 languages**, per-language dedup + quality filter.

### Data mix (Sec. 3.1.2)

| Domain | Share |
|---|---|
| General knowledge | ~50% |
| Math & reasoning | ~25% |
| Code | ~17% |
| Multilingual | ~8% |

Picked via scaling-law experiments over candidate mixes.

### Annealing (Sec. 3.1.3)

- **Annealing** at the end of pretraining: 40M tokens, LR → 0, upsampled high-quality data. Polyak averaging of checkpoints = released weights.
- **As a data-evaluation tool** (see [annealing-as-data-eval](../pre-training/annealing-as-data-eval.md)): anneal a 50%-trained 8B on 40B tokens with 30% candidate + 70% default mix; reports **GSM8K +24.0%, MATH +6.4% on 8B**. Efficient candidate-dataset scoring.

---

## 4. Infrastructure

See [_parallelism](../pre-training/_parallelism.md) and [_communication-primitives](../systems/_communication-primitives.md).

### Hardware (Sec. 3.3.1)

- **16,000 H100 GPUs** (full cluster 24K, use 16K for 405B). Meta Grand Teton (8 GPUs + 2 CPUs per server), 700W TDP, 80GB HBM3.
- Network: **RoCE at 400 Gbps** for 405B (smaller models used InfiniBand). Three-layer Clos, 1:7 oversubscription at aggregation.
- Storage: Tectonic, 240 PB, 2 TB/s sustained.

### 4D parallelism (Sec. 3.3.2)

Order `[TP, CP, PP, DP]`:

1. **[Tensor parallelism](../pre-training/tensor-parallelism.md)** — TP=8 within server (NVLink).
2. **[Context parallelism](../pre-training/context-parallelism.md)** — CP=16 at 128K. Uses **all-gather CP** (Llama 3's variant), not Ring Attention; supports arbitrary masks including intra-document.
3. **[Pipeline parallelism](../pre-training/pipeline-parallelism.md)** — PP=16, modified interleaved 1F1B with tunable N, first/last stage rebalanced by moving one layer off each end. No DualPipe.
4. **[FSDP](../pre-training/fsdp.md)** (ZeRO-2-ish variant; `reshard_after_forward=False`) — DP varies by config.

Three Table 4 configurations for 405B:
- 8,192 GPUs, TP=8 CP=1 PP=16 DP=64, seq=8K: **MFU 43%**.
- 16,384 GPUs, TP=8 CP=1 PP=16 DP=128, seq=8K: **MFU 41%**.
- 16,384 GPUs, TP=8 CP=16 PP=16 DP=8, seq=131K: **MFU 38%**.

### Precision

**BF16 pretraining** with FP32 gradient accumulation + FP32 reduce-scatter. **No FP8** for training (used only for inference quantization). Explicit contrast with DeepSeek-V3's FP8-native training.

### Reliability (Sec. 3.3.4)

- **Effective training time >90%.**
- 54 days: **466 interruptions** (47 planned + 419 unexpected). 78% hardware; GPUs = 58.7% of all issues. **3 manual interventions** across the 54 days.
- Diurnal 1–2% throughput variation (ambient temperature DVFS).
- Tens-of-MW power-grid fluctuations during synchronized collective events.

---

## 5. Training recipe

Three stages (Sec. 3.4):

### Initial pretraining (Sec. 3.4.1)

- Peak LR 8e-5 (405B), linear warmup 8K steps, cosine decay to 8e-7 over 1,200,000 steps.
- Batch-size ramp: 4M (seq 4K) → 8M (seq 8K at 252M tokens) → 16M (seq 8K at 2.87T tokens).
- **"Few loss spikes; no divergence interventions."** Unusual for 405B scale.
- Mid-run mix adjustments: upsample non-English, math; add recent web data.

### Long-context pretraining (Sec. 3.4.2)

- Six stages from **8K → 128K**. Total ~800B tokens.
- Per-stage gate: short-context recovery + perfect needle-in-haystack at current length.
- Pure [RoPE](../fundamentals/rope.md) base scaling (θ = 500,000); no YaRN.

### Annealing (Sec. 3.4.3)

- 40M tokens, LR → 0, 128K context, upsampled high-quality data.
- **Polyak averaging** of checkpoints = released weights.

---

## 6. Post-training — 6-round iterative pipeline

See [reward-modeling](../post-training/reward-modeling.md), [dpo](../post-training/dpo.md), [rejection-sampling](../post-training/rejection-sampling.md), [capability-experts](../post-training/capability-experts.md), [knowledge-probe-hallucination](../post-training/knowledge-probe-hallucination.md).

### The 6 rounds

```
Round k:
    Train RM on ALL cumulative preference data
    Rejection-sample from round-(k-1)'s best checkpoint (or capability expert)
      K = 10 to 30 rollouts per prompt, top-1 via RM
    SFT on RS data + synthetic targeted data + small human
    DPO on MOST RECENT preference data (LR 1e-5, β = 0.1)
    Model-soup across hyperparameter/data variants at RM, SFT, DPO stages
```

Six iterations of this (Sec. 4.1.6). Each round's model becomes the teacher for the next.

### Reward modeling (Sec. 4.1.2)

- Initialized from pretrained checkpoint.
- **Bradley-Terry loss WITHOUT the Llama 2 margin term.**
- **Three-way edited preferences**: `edited > chosen > rejected`.
- **Concatenate prompt + all responses into one row**, shuffled — efficient training.
- Data filter: keep only "significantly better" and "better" pairs (top 2 of 4-level scale).

### SFT (Sec. 4.1.3)

- Data mix (Table 7): General English 52.66%, Code 14.89%, Reasoning+tools 21.19%, Exam-like 8.14%, Multilingual 3.01%, Long-context 0.11%.
- Mask prompt tokens, compute loss only on response tokens.
- LR 1e-5 for 405B; 8.5K–9K steps for largest models.
- Most data is model-generated (rejection-sampled or synthetic); exact human-vs-synthetic split undisclosed.

### Rejection sampling (Sec. 4.2.2)

- **K = 10 to 30** rollouts per prompt.
- **Single-RM argmax** (top-1 selection).
- Sampled from the best checkpoint of previous round — which may be a **capability expert** (code expert for code prompts, multilingual expert for multilingual).
- **PagedAttention + prefix-sharing** → >2× throughput.
- **NOT applied for tool use** (no observed benefit).

### DPO — Llama 3's variant (Sec. 4.1.4)

Two specific modifications on top of vanilla DPO:

**(a) Format-token masking.** Mask header + termination tokens from both chosen and rejected in the loss — prevents conflicting gradient signals on shared formatting tokens.

**(b) Auxiliary NLL loss on chosen.** Add `0.2 × L_NLL(chosen)` to the DPO loss. Prevents the absolute log-prob of chosen from decreasing.

```
L_total = L_DPO(β = 0.1) + 0.2 · L_NLL(chosen)
```

Cited from Pang et al. 2024 / Pal et al. 2024; Llama 3 is the first frontier-scale deployment.

**Why DPO, not PPO** (direct quote, Sec. 4.1.4): *"DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval."*

### Model souping (Sec. 4.1.5)

Average models across hyperparameter/data-variant runs at **all three stages — RM, SFT, DPO** — not just one. Cites Wortsman 2022, Izmailov 2019, Li 2022.

---

## 7. Capability experts (Sec. 4.3)

See [capability-experts](../post-training/capability-experts.md).

Branch pretraining partway; continue on a domain-heavy mix; use the expert to generate SFT data for the flagship.

### Code Expert (Sec. 4.3.1)

- Continue on **1T tokens, >85% code**. LCFT to 16K context. SFT + DPO on code.
- Generates **>2.7M synthetic code SFT examples** via three pipelines:
  1. **Execution-feedback** (~1M): generate + static analysis + unit tests + execution + self-correct on failure.
  2. **PL translation**: Python/C++ → TypeScript/PHP, validated by parse/compile/execute.
  3. **Backtranslation** (~1.2M): docs → code → self-verified.

### Multilingual Expert (Sec. 4.3.2)

- Continue on **90% multilingual mix**.
- Used for RS on 7 target languages (German, French, Italian, Portuguese, Hindi, Spanish, Thai).
- Multilingual SFT mix: 2.4% human, 44.2% other-NLP, 18.8% RS, 34.6% translated reasoning.

### Math & reasoning (Sec. 4.3.3)

- Step-wise traces with Llama 3 as verifier.
- **PRMs + ORMs** filter invalid intermediate steps.
- **MCTS with stepwise RMs** for hard prompts. (Interesting contrast with R1, which rejected PRMs and MCTS.)
- Interleave code + text (PAL-style) for execution-grounded reasoning.

### Long context (Sec. 4.3.4)

- Synthetic SFT data: (1) QA over 8K doc chunks, (2) hierarchical summarization, (3) long-context code reasoning.
- Length buckets: 16K / 32K / 64K / 128K.
- **Mix ratio: 0.1% long-context synthetic data** with short-context SFT.
- **DPO stays short-context only** — long-context capability preserved by SFT stage alone.

### Tool use (Sec. 4.3.5)

- Tools: **Brave Search, Python interpreter, Wolfram Alpha**.
- Python objects + JSON function schemas.
- **Message-level preferences** (not response-level).
- **No rejection sampling** (no benefit observed).
- Zero-shot function-calling data synthesized from real function defs mined from The Stack.

### Factuality (Sec. 4.3.6)

See [knowledge-probe-hallucination](../post-training/knowledge-probe-hallucination.md).

Knowledge-probe pipeline: detect questions where the model is **consistently informative but wrong**, teach a refusal. Automated with Llama 3 as judge; small labeled dataset for sensitive topics.

### Steerability (Sec. 4.3.7)

System-prompt adherence. **The one capability using all four stages — RM, RS, SFT, DPO.**

---

## 8. Safety

See [llama-guard](../safety/llama-guard.md), [prompt-guard](../safety/prompt-guard.md), [code-shield](../safety/code-shield.md), [rainbow-teaming](../safety/rainbow-teaming.md), [uplift-evaluation](../safety/uplift-evaluation.md).

### Taxonomy

MLCommons 13-category hazard + Code Interpreter Abuse (Sec. 5.4.7).

### Safety training (Sec. 5.4.3)

- Quality > quantity.
- Data: adversarial + borderline + synthetic (Rainbow Teaming).
- Refusal tone classifier + zero-shot rewriter.
- Safety DPO: **pairs near-orthogonal in embedding space** teach good-vs-bad distinction best.
- **Smaller models need higher safety-to-helpfulness ratio** than larger models.

### Safety sidecar models

- **[Llama Guard 3](../safety/llama-guard.md)** — fine-tuned Llama 3 8B safety classifier. ~50–86% VR reduction at +26–102% FRR.
- **[Prompt Guard](../safety/prompt-guard.md)** — mDeBERTa 86M for jailbreak/injection detection.
- **[Code Shield](../safety/code-shield.md)** — static analysis for insecure LLM-generated code, 7 languages.

### Risk evaluation (Sec. 5.4.5)

See [uplift-evaluation](../safety/uplift-evaluation.md).

- **CBRN uplift study**: 2-person teams, 6-hour scenarios, SME-judged plans. **No significant uplift.**
- **Cyber uplift**: 62 volunteers (31 expert, 31 novice). **Insignificant uplift.**
- **Autonomous attack agent**: failed to gain initial access on target machines across 20–23 runs.
- **Prompt injection susceptibility**: 405B = 21.7%, 70B = 26%, 8B = 19%. Higher than GPT-4 Turbo (17%), lower than Mixtral (35%).

---

## 9. Multimodal — compositional, not released

See [cross-attention-adapter](../multimodal/vision/cross-attention-adapter.md), [vit](../multimodal/vision/vit.md), [clip](../multimodal/vision/clip.md), [best-rq](../multimodal/audio/best-rq.md), [conformer](../multimodal/audio/conformer.md).

**Caveat** (Sec. 7 opening): *"our multimodal models are still under development and not yet ready for release."* The released Llama 3.1 is text-only. Vision variants (Llama 3.2 11B/90B) came later.

### Vision (Sec. 7)

- Image encoder: **ViT-H/14, 630M + 8 gated self-attn layers = 850M params**. MetaCLIP-style alignment objective.
- Integration: **Flamingo-style gated cross-attention** every 4th LLM layer. For 405B: **~100B cross-attention params**.
- Two-stage integration training: initial (~6B pairs) + annealing (~500M).
- Image encoder **unfrozen** during integration (differs from Flamingo).
- Vision DPO: reference model **EMA-updated every k steps** (not frozen forever) to handle distribution shift.

### Video

- 64 frames max. **Perceiver Resampler** merging 32 consecutive frames → 1. Video cross-attention every 4th image cross-attn.

### Speech (Sec. 8)

- Encoder: **Conformer, 1B params, 24 layers** (see [conformer](../multimodal/audio/conformer.md)).
- Pretrained via **BEST-RQ** (see [best-rq](../multimodal/audio/best-rq.md)) on 15M hours of unlabeled speech.
- **Direct token insertion** (not cross-attention): speech embeddings fed as LLM tokens wrapped in special boundary tokens.
- Integration is not cross-attention (unlike vision). LLM frozen during adapter training; 650K updates at LR 1e-4 (for 8B variant).

### Speech generation

- **LLM not fine-tuned for speech output.** Prosody Model + Text Normalization read 8B Llama 3's 16th-decoder-layer embeddings via cross-attention.
- Proprietary TTS for waveform.

---

## 10. Results (Sec. 5, Table 2)

Llama 3.1 Instruct headline numbers:

| Benchmark | 8B | 70B | 405B | GPT-4 (0125) | GPT-4o | Claude 3.5 Sonnet |
|---|---|---|---|---|---|---|
| [MMLU](../evaluation/mmlu.md) (5-shot) | 69.4 | 83.6 | 87.3 | 85.1 | 89.1 | 89.9 |
| [MMLU-Pro](../evaluation/mmlu-pro.md) (5-shot CoT) | 48.3 | 66.4 | 73.3 | 64.8 | 74.0 | **77.0** |
| [IFEval](../evaluation/ifeval.md) | 80.4 | 87.5 | **88.6** | 84.3 | 85.6 | 88.0 |
| [HumanEval](../evaluation/humaneval.md) (0-shot) | 72.6 | 80.5 | 89.0 | 86.6 | 90.2 | **92.0** |
| GSM8K (8-shot CoT) | 84.5 | 95.1 | **96.8** | 94.2 | 96.1 | 96.4 |
| [MATH-500](../evaluation/math500.md) (0-shot CoT) | 51.9 | 68.0 | 73.8 | 64.5 | **76.6** | 71.1 |
| [GPQA](../evaluation/gpqa.md) (0-shot CoT) | 32.8 | 46.7 | 51.1 | 41.4 | 53.6 | **59.4** |
| [BFCL](../evaluation/bfcl.md) (tool) | 76.1 | 84.8 | 88.5 | 88.3 | 80.5 | **90.2** |
| InfiniteBench En.MC | 65.1 | 78.2 | **83.4** | 72.1 | 82.5 | — |
| [MGSM](../evaluation/mgsm.md) | 68.9 | 86.9 | **91.6** | 85.9 | 90.5 | **91.6** |

**Llama 3.1 405B matches or beats GPT-4-0125 on most benchmarks**, trails GPT-4o and Claude 3.5 Sonnet on some (code, GPQA), beats Claude 3.5 on ARC-Challenge, MATH, MGSM. First open model at frontier parity.

### Human eval (Sec. 5.3, Fig. 17)

7000 prompts, 6 single-turn + 3 multi-turn capabilities. 10/30/60 easy/medium/hard.
- vs **GPT-4 (0125)**: on par overall; 405B **beats** on multi-turn reasoning and coding.
- vs **GPT-4o**: mixed.
- vs **Claude 3.5 Sonnet**: 405B beats on English multi-turn; trails on code and reasoning.

---

## 11. What's opaque

- Absolute counts of preference pairs, SFT examples, safety-tuning data.
- RM / DPO hyperparameters (LR, epochs, batch).
- AdamW β values for the main 405B.
- Gradient clipping threshold.
- Per-round composition deltas.
- 405B pretraining GPU-hours (only 3.8 × 10²⁵ FLOPs and MFU are given).
- Checkpoint frequency.
- Individual long-context-stage token counts (only "six stages, ~800B total").
- Global human-vs-synthetic data split.

---

## 12. Key takeaways

1. **Dense at frontier scale is feasible.** 405B dense with ~43% MFU at 8K context. Llama 3's bet on "managing complexity via dense" paid off.
2. **4D parallelism as a default.** `[TP=8, CP=16, PP=16, DP=8]` is a production-validated recipe for frontier-scale long-context training. See [_parallelism](../pre-training/_parallelism.md).
3. **All-gather CP beats Ring Attention for GQA models.** When K/V are small (GQA-8), all-gather is cheaper than ring-rotation and handles arbitrary masks cleanly.
4. **BF16 + FP32 gradient accumulation is still the production default.** DeepSeek-V3's FP8-native training is the outlier; Llama 3's BF16 approach is the conservative default.
5. **DPO beats PPO at scale** — less compute, better IFEval. Augment with NLL auxiliary loss and format-token masking for stability.
6. **Capability experts are the scalable path to domain SFT data.** >2.7M synthetic code examples from the code expert; similar machinery for multilingual.
7. **Model souping at all three post-training stages** (RM, SFT, DPO), not just one. Broad applicability.
8. **Knowledge-probe factuality** — automated "detect what the model doesn't know, teach it to refuse." Reusable methodology.
9. **Downstream scaling laws** (compute → NLL → accuracy) give actionable predictions, not just loss curves. See [downstream-scaling-laws](../pre-training/downstream-scaling-laws.md).
10. **Annealing as data evaluation** — cheap candidate-dataset scoring via a 50%-trained 8B + 40B-token anneal. See [annealing-as-data-eval](../pre-training/annealing-as-data-eval.md).
11. **Safety is a layered system.** Main-model refusal + Llama Guard (content) + Prompt Guard (attacks) + Code Shield (insecure code). Deployers combine.
12. **Uplift evaluation as pre-release check.** Meta's CBRN and cyber studies are the emerging standard for "does this model pose catastrophic risk?" See [uplift-evaluation](../safety/uplift-evaluation.md).

---

*Pairs well with:* the [DeepSeek-V3](deepseek-v3.md) case study (the contrasting MoE + FP8-native + DualPipe approach to the same scale problem) and the [Kimi k1.5](kimi-k1-5.md) / [DeepSeek-R1](deepseek-r1.md) case studies (the long-CoT RL branch of post-training that Llama 3's classical-DPO recipe doesn't address).
