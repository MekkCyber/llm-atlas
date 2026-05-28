# Case Study: Qwen2.5

*Alibaba's dense + MoE model family spanning 0.5B to 72B (dense) plus proprietary MoE variants (Turbo, Plus). The interesting story is not a single architectural breakthrough — it's a methodical scaling of data (7T → 18T tokens), a comprehensive post-training pipeline (SFT → DPO → GRPO), and strong results across the full size range. The report is notably less transparent than DeepSeek-V3 on infrastructure and architecture internals, but provides the most detailed open post-training recipe of the Qwen series.*

**Related concepts:** [grpo.md](./../post-training/grpo.md) · [dpo.md](./../post-training/dpo.md) · [rope.md](./../fundamentals/rope.md) · [dca.md](./../fundamentals/dca.md) · [bpe.md](./../fundamentals/bpe.md) · [qk-norm.md](./../architectures/qk-norm.md) · [multi-head-attention.md](./../architectures/multi-head-attention.md) · [rejection-sampling.md](./../post-training/rejection-sampling.md)

---

## What this is

**Qwen2.5**, released September 2024 by Alibaba / Qwen Team. arXiv 2412.15115. A family of decoder-only transformers in seven dense sizes (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B) plus two proprietary MoE variants (Qwen2.5-Turbo, Qwen2.5-Plus). All dense models are open-weight under Apache 2.0.

The paper's core claim: the 72B model outperforms most open and proprietary models at release, while the smaller sizes (7B, 14B, 32B) set new Pareto points for their class. The recipe rests on three pillars:

1. **Pre-training data scaled to 18T tokens** (from 7T in Qwen2) with aggressive quality filtering using Qwen2-Instruct as a scorer, domain rebalancing, and synthetic data injection for math/code.
2. **Three-stage post-training**: SFT on 1M+ examples → offline RL (DPO) on ~150K pairs → online RL (GRPO) with a multi-criteria reward model.
3. **Long-context engineering**: progressive RoPE extension via ABF + YARN + Dual Chunk Attention, reaching 128K for dense models and 1M tokens for Turbo.

Unlike DeepSeek-V3 which is a single monolithic model with deep systems innovations, Qwen2.5 is a **model family** — the engineering story is about scaling the same recipe cleanly across 0.5B–72B.

---

## Architecture at a glance

All dense models share the same base architecture, varying depth and width:

```
Shared design:
  - Decoder-only transformer
  - Grouped Query Attention (GQA)
  - SwiGLU activation
  - RoPE positional embeddings
  - QKV bias in attention
  - RMSNorm (pre-normalization)
  - Byte-level BPE tokenizer, 151,643 vocab

Model configs:
  0.5B:  24 layers, 14 Q-heads / 2 KV-heads,  tied embeddings,   32K ctx
  1.5B:  28 layers, 12 Q-heads / 2 KV-heads,  tied embeddings,   32K ctx
  3B:    36 layers, 16 Q-heads / 2 KV-heads,  tied embeddings,   32K ctx
  7B:    28 layers, 28 Q-heads / 4 KV-heads,  untied embeddings, 128K ctx
  14B:   48 layers, 40 Q-heads / 8 KV-heads,  untied embeddings, 128K ctx
  32B:   64 layers, 40 Q-heads / 8 KV-heads,  untied embeddings, 128K ctx
  72B:   80 layers, 64 Q-heads / 8 KV-heads,  untied embeddings, 128K ctx
```

Models ≤ 3B use tied embeddings (input/output share weights); 7B+ untie them. Models ≤ 3B are limited to 32K context / 8K generation; 7B+ support 128K context / 8K generation.

### MoE variants

Qwen2.5-Turbo and Qwen2.5-Plus replace standard FFN layers with MoE layers using fine-grained expert segmentation and shared expert routing (same design lineage as Qwen1.5-MoE). **Exact expert counts, activated parameters, and total parameters are not disclosed.** Qwen2.5-Turbo supports up to 1M-token context length.

### Tokenizer

Byte-level BPE (BBPE) with 151,643 regular tokens. Qwen2.5 expands control tokens from 3 → 22, including 2 new tokens for tool-use formatting.

---

## Pre-training

### Data: 7T → 18T tokens

The headline number: pre-training data scaled from 7T tokens (Qwen2) to **18T tokens**. No percentage-level breakdown is disclosed, but the paper describes qualitative composition changes:

- **Quality filtering**: Qwen2-Instruct models serve as data quality scorers, performing multi-dimensional analysis to evaluate and score training samples. This is a bootstrapping pattern — the previous generation's instruct model becomes the next generation's data filter.
- **Domain rebalancing**: Using Qwen2-Instruct as a classifier, they downsampled overrepresented domains (e-commerce, social media, entertainment) and upsampled underrepresented ones (technology, science, academic content).
- **Synthetic data**: Generated using Qwen2-72B-Instruct and Qwen2-Math-72B-Instruct for math, code, and knowledge domains. Filtered with proprietary reward models before inclusion.
- **Specialist data injection**: Math data from Qwen2.5-Math and code data from Qwen2.5-Coder datasets were folded into the general pre-training mixture.

### Decontamination

Training sequences are removed if the longest common subsequence (LCS) with any test sequence satisfies **both** $|\mathrm{LCS}| \ge 13$ tokens **and** $|\mathrm{LCS}| \ge 0.6 \times \min(|s_{\text{train}}|, |s_{\text{test}}|)$. A dual-threshold approach — the absolute threshold catches short contaminated snippets, the relative threshold catches proportionally-large overlaps.

### Scaling laws

They studied the relationship between model size (N), dataset size (D), optimal learning rate (μ_opt), and optimal batch size (B_opt) across dense models (44M–14B params) and MoE models (44M–1B activated params) on 0.8B–600B tokens. The scaling laws were used to predict hyperparameters for larger runs and to estimate performance parity between MoE and dense variants. **No formulas are disclosed** — the paper references the experiments but does not publish the scaling relationships.

### Context length training

Two-phase approach:

**Phase 1:** Train at 4,096-token context length (the initial pre-training phase).

**Phase 2:** Extend context via RoPE base frequency increase:
- Standard models: 4K → 32,768 tokens, RoPE base increased from 10,000 → 1,000,000 using ABF (Adjusted Base Frequency).
- At inference, YARN + Dual Chunk Attention (DCA) further extend to 128K.

**Qwen2.5-Turbo progressive extension** (4 stages):
1. 32,768 tokens
2. 65,536 tokens
3. 131,072 tokens
4. 262,144 tokens

Each stage uses RoPE base 10,000,000 and a mixture of 40% max-length sequences + 60% shorter sequences. At inference, sparse attention extends Turbo to 1M tokens.

---

## Post-training

The post-training pipeline has three stages, applied sequentially. This is one of the clearer openly-documented three-stage recipes (SFT → DPO → GRPO).

### Stage 1: Supervised Fine-Tuning (SFT)

**Scale:** 1M+ examples across 9 domains, trained for 2 epochs.

**Training config:**
- Sequence length: 32,768 tokens
- Learning rate: 7×10⁻⁶ → 7×10⁻⁷ (decaying schedule)
- Weight decay: 0.1
- Gradient norm clipping: 1.0

**The 9 SFT data domains:**

1. **Long-sequence generation** — Up to 8K-token outputs. Used back-translation to generate queries from long documents, filtered with Qwen2.
2. **Mathematics** — Chain-of-thought data from Qwen2.5-Math. Sources include public datasets, K-12 collections, and synthetic problems. Generated via rejection sampling + reward model scoring + annotated reference answers.
3. **Coding** — From Qwen2.5-Coder. Multiple language-specific agents collaborate in a framework covering ~40 programming languages. Quality validated via automated unit testing in a multilingual sandbox.
4. **Instruction following** — Validated with code-based verification: generated verification code + comprehensive unit tests, then rejection-sampled based on execution feedback.
5. **Structured data understanding** — Tabular QA, fact verification, error correction, structural understanding. Includes reasoning chains.
6. **Logical reasoning** — 70,000 new queries spanning multiple-choice, true/false, and open-ended formats. Covers deductive, inductive, analogical, causal, and statistical reasoning.
7. **Cross-lingual transfer** — Translation models convert high-resource language data to low-resource languages. Semantic alignment evaluation between multilingual responses.
8. **Robust system instruction** — Hundreds of general system prompts with consistency evaluation across different prompts.
9. **Response filtering** — A dedicated critic model + multi-agent collaborative scoring; only flawless responses are retained.

### Stage 2: Offline RL — DPO

**Algorithm:** Direct Preference Optimization (DPO).

**Scale:** ~150,000 preference pairs, trained for 1 epoch.

**Training config:**
- Optimizer: Online Merging Optimizer (Lu et al., 2024)
- Learning rate: 7×10⁻⁷

**Pair construction:**
- Positive examples: responses passing quality checks (execution feedback, answer matching).
- Negative examples: responses failing quality checks.
- Validated via human + automated review.
- Focus domains: mathematics, coding, instruction following, logical reasoning.

The Online Merging Optimizer is referenced as an external technique (Lu et al., 2024a) — no internal details are given. It likely helps prevent DPO's known tendency to degrade general capabilities while optimizing preferences.

### Stage 3: Online RL — GRPO

**Algorithm:** Group Relative Policy Optimization (GRPO) — the same algorithm used in DeepSeek-V3/R1.

**Reward model criteria** (multi-dimensional, not a single score):
- **Truthfulness**: factual accuracy, faithful context reflection
- **Helpfulness**: useful, engaging, educational, relevant content
- **Conciseness**: succinct, avoiding verbosity
- **Relevance**: direct alignment with user query
- **Harmlessness**: no illegal/immoral/harmful content
- **Debiasing**: gender, race, nationality, political neutrality

**Training config:**
- 8 responses sampled per query
- Global batch size: 2,048
- Samples per episode: 2,048
- Query prioritization: by response score variance (higher variance = more learning signal = higher priority)

**Reward model data:**
- Queries from open-source data + proprietary high-complexity queries
- Responses generated from SFT, DPO, and RL checkpoints at varying temperatures
- Preference pairs via human + automated labeling

The query prioritization by score variance is a practical insight — queries where the model is inconsistent (sometimes good, sometimes bad) provide the most signal for RL. Queries it always gets right or always gets wrong contribute less.

### Long-context post-training (Turbo-specific)

Two-stage SFT:
1. Short instructions only (≤32K tokens) — same data as other Qwen2.5 models
2. Hybrid: short (≤32K) + long (≤262K) instructions

RL stage uses only short instructions. Rationale: computational expense of long-context RL + scarcity of suitable long-context reward models.

---

## Evaluation snapshot

### Flagship: Qwen2.5-72B-Instruct

| Benchmark | Qwen2.5-72B-Instruct | GPT-4o | Llama-3.1-405B | Claude 3.5 Sonnet |
| --- | --- | --- | --- | --- |
| MMLU-Pro | 71.1 | — | — | — |
| MMLU-redux | 86.8 | — | — | — |
| LiveBench 0831 | 52.3 | — | — | — |
| MATH | 83.1 | — | — | — |
| GSM8K | 95.8 | — | — | — |
| HumanEval | 86.6 | — | — | — |
| MBPP | 88.2 | — | — | — |
| LiveCodeBench | 55.5 | — | — | — |
| IFEval | 84.1 | — | — | — |
| Arena-Hard | 81.2 | — | — | — |
| MT-Bench | 9.35 | — | — | — |

*Note: The paper compares against many models but competitor numbers vary by benchmark. The 72B matches or exceeds GPT-4o-mini and competes with Llama-3.1-405B (a 5.6× larger model) on most benchmarks.*

### Base model: Qwen2.5-72B

| Benchmark | Qwen2.5-72B | Llama-3.1-70B |
| --- | --- | --- |
| MMLU | 86.1 | ~79 |
| MMLU-Pro | 58.1 | — |
| BBH | 86.3 | — |
| GSM8K | 91.5 | — |
| MATH | 62.1 | — |
| HumanEval | 59.1 | — |
| MBPP | 84.7 | — |

### MoE variants vs. 72B dense

| Benchmark | Qwen2.5-72B-Instruct | Qwen2.5-Plus | Qwen2.5-Turbo |
| --- | --- | --- | --- |
| MMLU-Pro | 71.1 | 72.5 | 64.5 |
| MATH | 83.1 | 84.7 | 81.1 |
| GSM8K | 95.8 | 96.0 | 93.8 |
| HumanEval | 86.6 | 87.8 | 86.6 |
| IFEval | 84.1 | 86.3 | — |
| Arena-Hard | 81.2 | 81.4 | 67.1 |

Qwen2.5-Plus slightly outperforms the 72B dense model on most benchmarks, presumably with fewer activated parameters.

### Size-class benchmarks (instruct models)

| Benchmark | 0.5B | 1.5B | 3B | 7B | 14B | 32B | 72B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MMLU-Pro | 15.0 | 32.4 | 43.7 | 56.3 | 63.7 | 69.0 | 71.1 |
| MATH | 34.4 | 55.2 | 65.9 | 75.5 | 80.0 | 83.1 | 83.1 |
| GSM8K | 49.6 | 73.2 | 86.7 | 91.6 | 94.8 | 95.9 | 95.8 |
| HumanEval | 35.4 | 61.6 | 74.4 | 84.8 | 83.5 | 88.4 | 86.6 |

Notable: the 3B matches or exceeds many 7B-class models from the previous generation, and the 32B nearly matches the 72B on several benchmarks.

### Long-context: RULER benchmark

| Context | 7B | 14B | 32B | 72B | Turbo |
| --- | --- | --- | --- | --- | --- |
| 4K | 85.4 | 91.4 | 92.9 | 95.1 | 93.1 |
| 32K | 93.7 | 95.9 | 95.5 | 97.7 | 95.5 |
| 64K | 89.4 | 93.4 | 95.5 | 96.5 | 94.8 |
| 128K | 82.3 | 86.7 | 90.3 | 93.0 | 90.8 |

Without DCA+YARN, 72B drops from 93.0 → 67.0 at 128K. Turbo achieves 100% passkey retrieval accuracy at 1M tokens.

### Reward model: Qwen2.5-RM-72B

| Benchmark | Score |
| --- | --- |
| RewardBench (Overall) | 91.59 |
| RewardBench Chat | 97.21 |
| RewardBench Safety | 92.71 |
| RewardBench Reasoning | 97.65 |

---

## Sparse attention for 1M context (Turbo)

Qwen2.5-Turbo uses a sparse attention mechanism at inference to handle 1M-token sequences. The paper claims:
- **12.5× reduction** in attention computation for 1M-token sequences
- **3.2–4.3× speedup** in time-to-first-token (TTFT) across hardware configurations
- No algorithmic details are disclosed — the sparsity pattern is not described.

---

## What's interesting

1. **Previous-gen-as-data-filter is a flywheel.** Qwen2-Instruct scores training data for Qwen2.5, and Qwen2.5 will presumably score data for Qwen3. This is an iterative quality ratchet — each generation's instruct model bootstraps the next generation's data pipeline.

2. **Three-stage post-training (SFT → DPO → GRPO) is the most complete open recipe.** Most papers describe one or two stages. Qwen2.5 provides concrete details on all three in sequence, including the rationale for the DPO → GRPO ordering: DPO provides a strong initialized policy for GRPO to refine with online exploration.

3. **Query prioritization by score variance in GRPO.** Simple but effective — spend RL compute on queries where the model is inconsistent, not on things it always gets right or always fails. This is a curriculum-learning insight applied to RL.

4. **Code-validated instruction following.** They generate verification code with unit tests for instruction-following data, then rejection-sample based on execution feedback. This brings the rigor of code benchmarks to open-ended instruction data.

5. **The 32B is suspiciously close to the 72B.** On MATH both score 83.1; on GSM8K the 32B is 95.9 vs 72B's 95.8. This suggests diminishing returns at the top of the dense scaling curve for this training recipe, or that post-training equalizes models that are "large enough."

6. **Tied embeddings at ≤3B, untied at 7B+.** A practical design choice: tying saves significant parameter budget when the embedding matrix is a large fraction of total params (as it is for small models). At 7B+ the FFN/attention parameters dominate and untying is worth the memory.

---

## What's opaque

- **Training infrastructure**: no GPU counts, training time, or cost disclosed. Compare to DeepSeek-V3's explicit $5.576M figure.
- **MoE architecture**: expert counts, total parameters, and activated parameters for Turbo and Plus are not disclosed.
- **Pre-training data composition**: no percentage breakdown by domain. We know it's 18T tokens and which domains were up/down-sampled, but not the mixture ratios.
- **Scaling law formulas**: experiments described but no equations published.
- **Sparse attention mechanism**: claimed 12.5× reduction but the sparsity pattern is not described.
- **Online Merging Optimizer**: referenced but not explained — critical to the DPO stage but readers must chase the citation.
- **Reward model architecture**: evaluation published but training details (architecture, data scale, loss function) are minimal.

The overall transparency level is moderate — more open than GPT-4 or Gemini reports, less open than DeepSeek-V3 or OLMo 2. The post-training section is the most detailed part; the pre-training and infrastructure sections are where information is withheld.

---

## Key takeaways

1. **Data scaling + quality filtering > architectural novelty.** Qwen2.5 uses no new architectural ideas over Qwen2 — same GQA, same SwiGLU, same RoPE. The gains come from 2.6× more data that's better filtered and domain-rebalanced. The architecture is standard; the data is the moat.

2. **SFT → DPO → GRPO is a validated three-stage recipe.** The DPO stage initializes a better policy for GRPO to work with. The paper demonstrates this works across the full 0.5B–72B range.

3. **Multi-criteria reward models outperform single-score approaches.** Evaluating truthfulness, helpfulness, conciseness, relevance, harmlessness, and debiasing independently gives more nuanced RL signal than a scalar preference score.

4. **Long-context is an inference-time problem, not a training-time problem.** Train at 4K, extend to 32K via RoPE adjustment, reach 128K–1M via inference-time techniques (YARN, DCA, sparse attention). The training cost stays manageable while serving cost scales with user demand.

5. **Model family coherence matters.** A single training recipe that works from 0.5B to 72B with predictable quality scaling is operationally more valuable than a single flagship model. Users can pick the cost/quality point that fits.

6. **The 32B sweet spot.** Matching the 72B on key benchmarks at less than half the parameters makes 32B the most interesting model in the family for deployment — similar quality, much cheaper inference.

---

*Pairs well with:* the [deepseek-v3.md](./deepseek-v3.md) for contrast — DeepSeek-V3 is a single frontier model with deep systems innovations (MLA, DualPipe, FP8, aux-loss-free MoE), while Qwen2.5 is a model family scaling a standard architecture through data and post-training quality. Different strategies, both reaching frontier performance. Also compare the [olmo-2.md](./olmo-2.md) for the opposite end of the transparency spectrum.
