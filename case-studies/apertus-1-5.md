# Case Study: Apertus 1.5

*A fully-open 8B / 70B Swiss-stack LLM upgraded from text (Apertus 1.0) to multimodal reasoning via continued pretraining, with 4× context extension to 262k, a multi-stage post-training recipe, and full release of weights, intermediate checkpoints, data pipelines, training code and eval harness under Apache 2.0.*

**Related concepts:** [discrete-multimodal-tokens](../multimodal/discrete-multimodal-tokens.md) · [sdpo](../post-training/sdpo.md) · [capability-filtering](../data/capability-filtering.md) · [mid-training](../pre-training/mid-training.md) · [rlvr](../post-training/rlvr.md) · [dpo](../post-training/dpo.md) · [rope](../fundamentals/rope.md) · [olmo-2 case study](olmo-2.md)

---

## What this is

**Apertus 1.5**, released October 2026 by a Swiss consortium (ETH Zurich / EPFL / HES-SO Valais-Wallis / FHNW / SDSC / CSCS / Univ. Zurich / Univ. Bern). Two artifacts:

1. **Apertus 1.5 8B.** 8B parameters, continued-pretrained from Apertus 1.0 8B on ~4T additional tokens; adds image + audio understanding, 262,144-token context, thinking mode, and improved instruction-following and tool use.
2. **Apertus 1.5 70B.** Same recipe, ~2T additional tokens, ~9× the parameter count of 8B.

The paper is a full tech report comparable to OLMo-2 in openness: all data, code, checkpoints, and eval harnesses are released. It is also one of the first truly open multimodal reasoners at this scale.

The key architectural idea: **multimodality enters through the tokenizer**, not through a visual encoder stapled on. Image patches and audio frames are quantized into **discrete codes** that occupy added rows in the embedding matrix; the LM head remains text-only, so the model consumes multimodal inputs but still generates text. See [discrete-multimodal-tokens](../multimodal/discrete-multimodal-tokens.md).

---

## Architecture at a glance

- **Scale:** 8B and 70B dense Transformers (hidden/layer/head details inherit from Apertus 1.0 and are not re-specified in the headline sections).
- **Tokenizer:** 131,072 original text entries + 200 reserved multimodal slots + 131,072 image codes + 4,096 audio codes = 266,440 entries. The released embedding matrix is 266,752 rows (312 padding).
- **Output head:** restricted to the **original text vocabulary** — the model does not generate multimodal codes.
- **Context length:** extended **8,192 → 32,768 → 65,536 → 131,072 → 262,144** across staged phases. RoPE base frequency raised from 500,000 (scaling factor 8) to 4,000,000 (scaling factor 32).
- **Multimodal:** image + audio inputs encoded as discrete tokens from the expanded vocabulary; structure markers plus content codes.

---

## Training recipe

```
Apertus 1.0 (text only)
     │
     ▼
 Stage 1 — Continued pretraining on high-quality text
     │     (dclm-edu 47.5%, FineWeb2-HQ-Plus top-10% 20.1%, MegaMath 6.1%,
     │      Nemotron-Pretraining-Code 5.3%, …)
     ▼
 Stage 2 — Multimodal continued pretraining
     │     (text + image codes + audio codes;
     │      per-class token budgets for the long-context phase, 8B:
     │       English 6.07B · multilingual 4.05B · image 6.66B · audio 1.30B · SFT 5.60B)
     ▼
 Stage 3 — Math + code emphasis
     │     (dclm-edu 20.5%, Nemotron-Pretraining-Code 15.1%, MegaMath 9.5%,
     │      finemath 6.3%; Swiss caselaw 2.7%; Apertus 1.5 SFT 1.2%)
     ▼
 Context extension (staged)
     │     8k → 32k → 65k → 131k → 262k
     │     each step: previous checkpoint + optimizer reset
     │     LR held at final cooldown value
     │     batch size halved when seq length doubles (tokens/batch constant)
     │     data mix: cooldown subsample + long documents + synthetic verifiable
     │       tasks (CWE, DocOrder, artificial needles)
     │     cross-document masking + best-fit-decreasing packing
     ▼
 Post-training (see below)
     ▼
 Apertus 1.5 8B / 70B
```

### Continued pretraining details

- Stages 1 and 2 run on web text (dclm-edu dominates). Stage 3 cuts general web and raises math + code, with regional additions (Swiss caselaw).
- Multimodal tokens enter in Stage 2; the LM loss remains over text positions only.
- Reported token budgets: ~4T (8B), ~2T (70B).

### Context extension recipe

Four doublings — 8k → 32k → 65k → 131k → 262k — each with:

- Optimizer state reset at the start of the step.
- Learning rate pinned at the Apertus 1.0 final-cooldown value (no re-warmup).
- Per-step batch-sequence count halved so total tokens/batch stays constant.
- Mixed real long documents + synthetic verifiable tasks (CWE: Context-Within-Elements; DocOrder; artificial needles).
- Cross-document masking so unrelated packed documents don't bleed attention.
- RoPE base scaled 500k → 4M across the schedule.

### Post-training stages

Pipeline:

```
Base (post-pretraining)
   ↓
Production SFT               — ~3.7M text samples;
   ↓                           capability-filtering decides which prompts
                               receive verified reasoning traces
                               (see capability-filtering.md).
                               LRs: 8B = 6e-5, 70B = 1e-5.
Long-context SFT             — trains at 262k context to recover retrieval
   ↓                           (short-context SFT dropped RULER@262k from
                               54.8 → 8.3; long-ctx SFT recovers it).
RL (RLVR)                    — verifiable rewards only (math checker,
   ↓                           IF checker, code tests, exact match).
                               8B: 480 steps · 70B: 180 steps.
                               No length penalty.
DPO                          — offline DPO on 70B;
   ↓                           online DPO on both (MaxMin pairs,
                               Qwen3.6-27B judge).
SDPO charter alignment       — 8B only; 70B ships the DPO checkpoint
                               (see sdpo.md).
```

### Data engineering notes

- **Capability filtering** (SFT): no-trace prompts kept if solved pass@8; the rest get verified reasoning traces. See [capability-filtering.md](../data/capability-filtering.md).
- **Variety-aware back-translation** for Romansh (one of Switzerland's low-resource official languages).
- **Chat-format discipline:** explicit empty system block so user content cannot occupy the system slot; thinking delimiters kept as non-special tokens so they survive detokenization; tool-output delimiters masked from the loss.
- **Masking of injected documents** in DocOrder and needle tasks so the model doesn't train on the injected content directly.

---

## Headline results

### 8B

- Development-suite average: **49.3% → 54.8%** across preference alignment.
- IFEval: 80.2 (post-SFT) → **89.6** (post-RL + online DPO).
- Multi-IF: 74.0 → **84.5**.
- Degenerate-output rate: 1.7% → **0.6%**.

### 70B

- Development-suite average: **55.3% → 61.2%** across preference alignment.
- Degenerate-output rate: 10.6% → **3.3%**.
- SDPO charter-alignment uplift: **+2.4%**.

### Interesting ablations

- **Short-context SFT catastrophic forgetting of retrieval.** RULER@262k drops from 54.8 → 8.3 after short-context SFT; long-context SFT recovers.
- **SDPO vs self-reflection feedback.** SDPO win rate vs the DPO checkpoint: **73.2% vs 67.4%**.
- **Trace supervision cost.** On thinking-off vision tasks, trace supervision costs 2.2 points — a measured alignment-vs-capability tradeoff.

---

## What's individually novel (and deserves its own depth file)

Innovations the paper treats as its own, extracted from the recipe:

1. **Discrete-token multimodality** — image + audio codes added as additional vocabulary; LM head stays text-only. See [discrete-multimodal-tokens](../multimodal/discrete-multimodal-tokens.md).
2. **SDPO (charter-conditioned Self-DPO)** — alignment pass with a judge that scores against a published behavioral charter. See [sdpo](../post-training/sdpo.md).
3. **Capability filtering for SFT CoT** — prompts solvable at pass@8 skip reasoning-trace supervision; others receive verified traces. See [capability-filtering](../data/capability-filtering.md).

The staged context extension (8k→32k→65k→131k→262k with cooled LR, batch-sequence halving, RoPE base scaling 500k→4M, cross-document masking, synthetic verifiable long-context tasks) is a well-documented, fully-released recipe — a strong reference even where individual pieces are not novel.

---

## Key takeaways

1. **Multimodality via tokenizer expansion is a viable alternative to adapter stacks.** Reusing the Transformer backbone and absorbing modalities into the vocabulary is simpler than cross-modal projections; the restriction to text output is a deliberate design choice, not a limitation to work around.
2. **Staged context extension is a reproducible recipe.** Full doubling schedule, cooled LR, synthetic verifiable long-context data, cross-document masking — a reference implementation now exists in the open.
3. **Short-context SFT breaks long context.** The RULER@262k regression after short-context SFT is a concrete, surprising failure. Long-context SFT must follow context extension or capability is lost.
4. **Capability filtering is cheap and effective.** Not every SFT prompt needs a reasoning trace; only the hard ones do. The pass@8 cutoff is a well-defined selection rule.
5. **Charter-conditioned alignment (SDPO) measurably improves over bare DPO.** 73.2% win rate vs 67.4% for self-reflection feedback.
6. **Fully open at frontier adjacency.** With Apache-2.0 weights, data, code, intermediate checkpoints, and eval harness, Apertus 1.5 is one of the strongest open references for the 2026 multimodal reasoner recipe.

---

## What's still opaque

- **Exact hidden/layer/head dims** inherit from Apertus 1.0 and are not restated in Apertus 1.5's main text.
- **Training compute** (GPU-hours) is not stated in the public abstract and introduction.
- **Context-extension token totals** show inconsistencies between prose (50–63B/step) and Table 5 (18–24B/step) in the version read.
- **Audio tokenizer internals** (how 4,096 audio codes are produced) are not covered in the sections read.
- **70B SDPO result.** 70B ships the DPO checkpoint rather than an SDPO one; the paper doesn't fully quantify the SDPO gap at 70B.
