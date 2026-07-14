# Case Study: Soofi S 30B-A3B

*A sovereign, fully-open **hybrid Mamba-2 + MoE Transformer** foundation model built end-to-end on the German Industrial AI Cloud. 31.6B total / 3.2B active per token, trained on ~27T tokens with deliberately up-weighted German. Matches dense 14–27B models on aggregate benchmarks, dominates the "fully open" tier ahead of Olmo 3 32B and Apertus 70B, and gets a near-constant KV cache from the hybrid backbone.*

**Related concepts:** [_moe.md](../architectures/_moe.md) · [deepseek-moe.md](../architectures/deepseek-moe.md) · [aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md) · [mid-training.md](../pre-training/mid-training.md) · [wsd-schedule.md](../pre-training/wsd-schedule.md) · [_data-curation.md](../data/_data-curation.md) · [dolma.md](../data/dolma.md) · [olmo-2.md](./olmo-2.md)

---

## What this is

Soofi S 30B-A3B is a European sovereign foundation model released by a consortium anchored by **KI Bundesverband, DFKI, Fraunhofer IAIS/IIS**, and multiple German universities. Trained entirely on the **German Industrial AI Cloud** (operated by Deutsche Telekom in Munich, ~253K B200 GPU-hours between March and May 2026), it is released with weights, per-source data accounting, intermediate checkpoints, hyperparameters, and training/eval code under highly permissive terms.

The interesting parts:

1. It is a **hybrid Mamba-2 + MoE transformer** at 30B scale — the first fully-open release combining SSM sequence mixing with sparse-MoE FFN at this size class.
2. It ships a **per-source data accounting** — every source is either fully released, released under a permissive derivative, or documented at aggregate + exact-mixture level for commercial-licensed data.
3. **Sovereign infrastructure**: no US cloud dependency for the training. This is the political/logistic story that makes it a case study distinct from Olmo or DeepSeek.

Read this to see how the "fully open" reproducibility bar Olmo 2 pushed extends to hybrid MoE architectures under sovereign infrastructure constraints.

---

## Architecture — hybrid Mamba-2 + granular MoE

```
52 layers, model dim d = 2688
─────────────────────────────────
23 × Mamba-2 sequence-mixing layers    (state dim = 128)
23 × granular MoE FFN layers           (128 routed experts, top-6 + 2 shared;
                                        expert intermediate dim = 1856)
 6 × GQA attention layers              (the "attention islands")
─────────────────────────────────
Total params      ~31.6B
Active per token  ~3.2B  (~3.6B including embeddings)
```

The block schedule interleaves the three types. Only ~12% of the layers are full attention — the SSM (Mamba-2) layers do the bulk of sequence mixing, giving the near-constant inference cache growth that makes the model attractive for long-context, high-concurrency serving.

The MoE side is **granular** in the DeepSeekMoE sense: 128 relatively small experts with 6 activated per token, plus 2 shared experts always on. This is the same "fine-grained + shared" pattern used by DeepSeek-V3 (see [deepseek-moe.md](../architectures/deepseek-moe.md)), applied here in a smaller expert configuration to fit the 3B-active budget.

The hybrid Mamba-2 + attention interleaving is not novel to Soofi S — Jamba, Zamba, and Nemotron-H established the pattern earlier. Soofi S is the first *fully open, per-source-audited* release at this scale that adopts it.

---

## Training data — two-phase, German-upweighted

Two phases with different mixtures; German weight jumps in Phase 2.

### Phase 1 — breadth (~20T tokens)

| Source | Share |
|---|---|
| English Web | 50.3% |
| Code | 14.6% |
| Reasoning | 10.4% |
| Academic / Wiki | 8.0% |
| German | 7.2% |
| Mathematics | 6.0% |
| SFT (mixed in) | 3.5% |

### Phase 2 — [mid-training](../pre-training/mid-training.md) (~6.6T tokens)

| Source | Share |
|---|---|
| English Web | 36.4% |
| Code | 16.4% |
| Reasoning | 11.8% |
| SFT | 8.5% |
| Academic / Wiki | 6.4% |
| Mathematics | 5.2% |
| **German** | **15.3%** |

German representation more than doubles in Phase 2 (7.2% → 15.3%). This is the "spend the late-stage tokens on what you want the model to be good at" pattern — same shape as Olmo 2's Dolmino Stage 2 late-lift on math/code.

Total training corpus: **~27 trillion tokens**.

---

## Training recipe

### Optimizer + schedule

- **[WSD schedule](../pre-training/wsd-schedule.md)** — Warmup–Stable–Decay. Peak LR `1e-3`, decayed to `1e-5`.
- Warmup: **254 iterations** for base pretraining; **100 iterations** for the long-context phase.
- **Global batch = 3072 sequences × 8192 tokens = 25.17M tokens per step.**
- **bf16 mixed precision** (not FP8 — B200 supports FP8 but the release does not disclose using it at training time).

### Infrastructure

- **512 × NVIDIA B200 GPUs** (64 × DGX B200 nodes of 8 GPUs).
- Deutsche Telekom's **Industrial AI Cloud** in Munich.
- **8-rail NVIDIA Quantum-2 NDR InfiniBand**, one 400 Gb/s ConnectX-7 port per GPU.
- Total wall clock: **~50 days** (24 March 2026 → 13 May 2026). ~253K B200 GPU-hours.

512 B200s for ~50 days is a mid-sized frontier training footprint. The design choice (30B-A3B, hybrid, mid-2026 hardware) targets the point where a single sovereign cluster can produce a fully-open competitive model.

---

## Post-training

Details are lighter in the released material at digest time, but the release includes SFT data mixed into pretraining Phase 2 (8.5% share) — an "SFT-in-pretraining" pattern (similar in spirit to some Qwen-style pipelines) rather than a strict SFT-then-DPO-then-RL post-training stack. Full post-training accounting is expected in the technical report follow-up.

---

## Evaluation snapshot

Selected comparisons drawn from the report's evaluation tables.

| Benchmark set | Soofi S | Olmo 3 32B | Apertus 70B |
|---|---|---|---|
| English aggregate | **70.1** | 67.3 | — |
| German aggregate | **79.1** | 69.2 | 72.8 |
| HumanEval | **73.8** | 63.0 | — |
| MBPP-DE (German code) | **84.2** | 70.8 | — |

Soofi S also **matches dense 14–27B models** (Gemma 3, Ministral) on aggregate English and German benchmarks — while activating only ~3.2B per token.

Serving efficiency: **~8–9× decode TPS/GPU at 40K context** vs dense 14–24B baselines, coming from the hybrid backbone's near-constant KV/state.

---

## What's actually released

| Artifact | Soofi S |
|---|---|
| Base model weights | ✓ |
| **Selected intermediate checkpoints** | ✓ |
| Per-source data accounting | ✓ (with commercial-source mixture disclosed) |
| Training + eval code | ✓ |
| Hyperparameters, LR schedule, batch config | ✓ |
| Data-construction artifacts for permissive sources | ✓ |

Not a strict Olmo-style *every-step* checkpoint stream, but comparable to the practical reproducibility bar Olmo 2 set. The differentiator is **per-source data accounting** including the commercial-licensed slice, which Olmo does not disclose at that granularity.

---

## Key takeaways

1. **Hybrid Mamba-2 + MoE is production-viable at 30B.** Interleaving SSM layers with a few GQA attention "islands" plus granular MoE FFN gives Soofi S a near-constant inference cache and dense-14–27B quality at 3.2B active. If replicated, this is the new default architecture for cost-sensitive open-source releases at this size.

2. **Sovereign infrastructure is now capable of frontier-adjacent open releases.** 512 B200s in Munich for 50 days is *enough*, given the right architecture choice, to compete with the fully-open frontier. Not GPT-5-class, but Olmo/Apertus-class — a real political outcome.

3. **Per-source data accounting is achievable even with commercial data.** Soofi S releases exact-mixture stats and aggregate volume for commercial-licensed sources rather than skipping them entirely. A middle path between "everything must be permissive" (limits quality) and "opaque mix" (Olmo already improved on this; Soofi S goes further).

4. **Two-phase pretraining with late-stage locale upweighting is now a template.** German share more than doubles in Phase 2 (7.2% → 15.3%). Same recipe shape as Olmo 2's Dolmino for math/code — spend the last ~25% of tokens on the axis you want to shine.

5. **WSD schedules keep winning.** No cosine drama; warmup, stable at peak LR, then decay. Matches the trend across recent open releases.

---

## What's still opaque at digest time

- **FP8 training**: B200s support FP8 but the disclosed hyperparameters say bf16. Unclear whether the team evaluated FP8 and rolled back, or never tried it.
- **Full post-training breakdown**: SFT-in-pretraining is disclosed; DPO/RL post-training details are lighter and expected in the follow-up technical report.
- **Tokenizer**: vocab size and byte-BPE parameters not stated in the release notes.
- **Ablation attribution**: the report claims the hybrid + MoE combination is the source of the serving-efficiency win, but per-component ablations (dense vs hybrid, MoE vs dense-FFN, granular vs coarse experts) are not itemized at digest time.

---

*Pairs well with:* the [Olmo 2 case study](./olmo-2.md) for the "fully open" reference on dense transformers — Soofi S extends Olmo 2's reproducibility bar to hybrid MoE + sovereign infrastructure.
