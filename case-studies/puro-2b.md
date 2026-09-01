# Case Study: Puro-2B

*Tsinghua PACMAN's fully-open Qwen2-1.5B-class model, pretrained from scratch on **consumer RTX 5090 GPUs** for under **$6.9K** — an academic-scale pretraining recipe that ships the whole pipeline (data, code, weights, ablations) under Apache 2.0.*

**Related concepts:** [fp8-training](../pre-training/fp8-training.md) · [fp8](../quantization/fp8.md) · [wsd-schedule](../pre-training/wsd-schedule.md) · [model-souping](../pre-training/model-souping.md) · [mid-training](../pre-training/mid-training.md) · [curriculum-model-averaging](../pre-training/curriculum-model-averaging.md) · [puro-cost-scaling-law](../pre-training/puro-cost-scaling-law.md)

---

## What this is

The interesting contribution is not the final model. It's the demonstration that a **~1.5B-parameter LLM competitive with Qwen2.5-1.5B** can be pretrained from scratch by a single lab on **consumer GPUs** under a **five-figure compute budget**, with every piece of the pipeline released. Prior open recipes existed but sat at $700K (SmolLM3-3B) to $1.5M+ (Llama-3.2-3B) — orders of magnitude out of reach for most academic and open-source groups.

The report combines a hardware choice (RTX 5090), a low-precision regime (FP8), an optimizer ("hyperball"), a checkpoint-averaging trick along the data curriculum, and a curated data recipe. It also derives a **cost scaling law** that predicts the reachable performance frontier as a function of dollars spent.

Read this report if you want a concrete existence proof that ground-up pretraining at the 1–3B scale is back within reach of small labs, and a template for what levers matter.

---

## The cost frontier being attacked

| Recipe | Cost | Params | Tokens |
|---|---|---|---|
| Llama-3.2-3B (from scratch) | ~$1.5M | 3B | ~15T |
| SmolLM3-3B (reproduction) | ~$700K | 3B | ~10T |
| **Puro-2B (best)** | **<$6.9K** | 1.5B | up to 1.4T |
| Puro Cost Law extrapolation | ~$4.4K (< $5,090) | 1.5B | — |

The Puro Cost Scaling Law, fit to the Puro-2B collection, predicts that **$4.4K** is sufficient to reach Qwen2-1.5B performance — hence the branding of "$5090" being both the GPU model *and* the training budget cap.

---

## Architecture — a compact, familiar transformer

Architectural choices are deliberately conservative. The novelty is *how* it's trained, not *what* it is: a decoder-only transformer in the 1.5B-parameter Qwen2-class shape (roughly ~28 layers, ~1600 hidden, GQA, SwiGLU, RoPE, RMSNorm — same family as Qwen2-1.5B). Ships as a *collection* of checkpoints that differ in **token budget** and **recipe variant** so the cost scaling law can be fit.

The report explicitly avoids exotic architectural changes — the point is that a stack of well-known cost levers, applied together on standard architecture, is enough.

---

## The cost levers (stacked)

Puro-2B stacks five levers, each individually documented but rarely combined:

### 1. Consumer-hardware selection

**RTX 5090** instead of H100/A100. Effective cost per training-token drops sharply once you leave datacenter GPUs — the tradeoff is memory per card, interconnect, and reliability. The report shows the tradeoff is favourable at the 1.5B parameter scale when the rest of the recipe respects the hardware.

### 2. [FP8 training](../pre-training/fp8-training.md)

FP8 mixed-precision throughout pretraining, following the modern recipe: FP8 in matmuls, higher-precision master weights and reductions, careful scaling. Roughly 2× throughput vs bf16 on supported hardware. Combined with a consumer GPU this is the biggest single cost lever.

### 3. Hyperball optimization

A geometry-aware optimizer applied to pretraining. The report describes it as a normalized-step-length optimizer that respects a "hyperball" constraint on updates — details are limited without the full text. Documented at the case-study level; a dedicated depth file will follow once the optimizer's mechanism is stable and reproduced.

### 4. [Curriculum model averaging](../pre-training/curriculum-model-averaging.md)

Model averaging done **along the data-curriculum trajectory** — average checkpoints taken at different stages of the pretraining curriculum, not just late-stage runs from the same seed as in classical model souping. Free quality on top of any run.

### 5. Data recipe

Curated pretraining mix chosen to fit the token budget. The report includes a **controlled ablation on pretraining data curricula** — how they shape downstream behaviour *after post-training* — which is exactly the kind of controlled study only a fully-open pipeline can support.

---

## Training — token budgets and the Puro collection

The Puro-2B **collection** is a family of checkpoints spanning:

- **Token budgets**: from small (< 500B tokens) up to **~1.4T tokens** for the best model.
- **Recipe variants**: subsets of the levers above, plus alternative data mixes and optimizer settings.

This grid is what powers the Puro Cost Scaling Law: with performance measured per checkpoint and cost known per token, they fit a curve of expected average performance as a function of dollars spent.

**LR schedule.** Practical short-budget training under this cost regime favours [WSD-style schedules](../pre-training/wsd-schedule.md) (warmup–stable–decay), which give a defensible way to stop early without wasted decay.

---

## Post-training

Optional post-training (SFT + preference optimization) is applied on top of selected base checkpoints for downstream evaluation. The **individually-novel** piece here is not the post-training recipe itself but the **controlled study of how the pretraining curriculum shapes what post-training can achieve** — enabled by having access to every checkpoint in the pretraining trajectory.

---

## Results snapshot

| Model | Compute cost | Approx. performance |
|---|---|---|
| Puro-2B (best) | < $6.9K | approaches Qwen2.5-1.5B under the paper's eval protocol |
| Puro Cost Law extrapolation to Qwen2-1.5B parity | ~$4.4K | matches Qwen2-1.5B |
| SmolLM3-3B (open reproduction) | ~$700K | 3B-class baseline |
| Llama-3.2-3B (from scratch) | ~$1.5M | 3B-class baseline |

Numbers approximate; the report tables carry the exact eval harness and per-checkpoint scores.

---

## The Puro Cost Scaling Law

Not a Chinchilla-style compute-optimal curve (parameters vs tokens), but a **cost-side** law: expected average performance as a function of **dollars of training cost**, once hardware and precision are fixed. Fit across the Puro-2B collection. Its concrete prediction — $4.4K to reach Qwen2-1.5B parity — is the sharp side of the report.

See [puro-cost-scaling-law.md](../pre-training/puro-cost-scaling-law.md) for the form of the law and how to think about it alongside classical Chinchilla / Hoffmann-style scaling.

---

## What's actually released

| Artifact | Puro-2B has it |
|---|---|
| Model weights (Apache 2.0) | ✓ |
| Full data recipe | ✓ |
| Training code | ✓ |
| Every checkpoint in the collection | ✓ (grid over token budget × recipe variant) |
| Cost-law fitting scripts | ✓ |
| Post-training controlled-curriculum study | ✓ |

HuggingFace collection: `thu-pacman/puro-2b`.

---

## Key takeaways

1. **Ground-up pretraining at the 1–3B scale is back within reach of small labs.** The gap to Llama-3.2-3B closed from ~$1.5M to under $6.9K by *stacking* consumer hardware, FP8, geometry-aware optimization, curriculum averaging, and a tight data recipe.

2. **[FP8 training](../pre-training/fp8-training.md) + consumer GPUs is the dominant cost lever** at this scale. Neither in isolation matches the combined effect.

3. **[Curriculum model averaging](../pre-training/curriculum-model-averaging.md)** generalizes classical [model souping](../pre-training/model-souping.md) beyond same-basin same-stage averaging — averaging along the curriculum trajectory is a genuinely different regularizer.

4. **Cost-side scaling laws** ([puro-cost-scaling-law](../pre-training/puro-cost-scaling-law.md)) complement Chinchilla-style compute-optimality: they answer "given a dollar budget, what performance can I reach?" — the operational question for small labs.

5. **Full-pipeline openness enables controlled studies you can't otherwise run** — like measuring how pretraining data curricula shape post-training outcomes. Open weights alone aren't enough; you need the data and code too.

---

*Pairs well with:* [OLMo 2](./olmo-2.md) for the openness lens at 7B–32B scale, and [DeepSeek-V3](./deepseek-v3.md) for FP8 pretraining at 671B scale (the far end of the same lever).
