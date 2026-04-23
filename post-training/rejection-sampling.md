# Rejection Sampling (for SFT Data Generation)

*Depth — generate K candidate completions per prompt, keep the ones that pass a check, SFT on the survivors.*

**TL;DR:** At its simplest: for each prompt, sample `K` responses from a model, score each, and **keep only the ones that pass some bar** (the verifier says it's correct; a reward model puts it above threshold; a judge model approves). Then SFT on the kept set. It's the cheap way to turn a partially-capable model into better training data — the model generates its own labels, you filter, you train. Used as a **self-improvement loop** in Llama 3 post-training and as **Stage 3** of the DeepSeek-R1 pipeline (the 800k-sample resample that turns R1-Zero-style RL traces into a legible SFT dataset).

**Prereqs:** [_post-training](_post-training.md), [rlvr](rlvr.md)
**Related:** [grpo](grpo.md), [orm](reasoning/orm.md), [long2short](reasoning/long2short.md), [deepseek-r1 case study](../case-studies/deepseek-r1.md), [kimi-k1-5 case study](../case-studies/kimi-k1-5.md)

---

## What it is

Three widely-used data-generation patterns for SFT:

| Pattern | Source of labels |
| --- | --- |
| Human SFT | Humans write responses |
| Distillation SFT | A larger/stronger model writes responses |
| **Rejection-sampling SFT** | The model (or a variant of it) writes **many** responses; a filter keeps only the good ones |

Rejection sampling is the middle ground when humans are expensive and no stronger model is available. You pay in *compute at inference* (sampling `K` responses per prompt) to save on *labeling*.

The two knobs that matter:

1. **The filter** — a verifier, a reward model, a judge LLM, or a combination. The filter is the bottleneck on quality: garbage filter ⇒ garbage training data.
2. **K** — how many rollouts per prompt. Bigger `K` gives higher yield (more prompts with at least one passing response) at linear cost.

The output is a filtered SFT dataset. You then run standard supervised fine-tuning on it.

---

## How it works

### The pipeline

```
for each prompt q in dataset:
    {o_1, ..., o_K} = sample K responses from generator(q)
    passing = [ o_i for o_i in responses if filter(q, o_i) ]
    if len(passing) > 0:
        keep one (or all) of passing as SFT target for q
    else:
        discard q

sft_dataset = [(q, kept_response) for q, kept_response in ...]
model.sft(sft_dataset)
```

`K` in the literature: 8 to 64 per prompt typical. `K = 16` is a common default.

### Filter types

| Filter | Use case | Cost |
| --- | --- | --- |
| **Rule-based verifier** | Math (answer match), code (tests pass), format | Free, deterministic |
| **Outcome reward model (ORM)** | Math/reasoning where verification is fuzzy | One RM forward pass per sample |
| **Preference reward model** | Helpfulness, style | One RM forward pass per sample |
| **Generative judge (LLM-as-judge)** | Open-ended correctness, "does this answer the question" | One LLM forward pass per sample; most expensive |
| **Composite** | Anything — e.g., "verifier passes AND length < 2k AND no language mixing" | Sum of above |

The right filter depends on the task. For math, a rule verifier beats everything: perfectly aligned, dirt cheap. For writing quality, you typically stack an RM with hard rules (length, formatting, refusal detection).

### When and how to "keep"

Two common choices once multiple rollouts pass:

- **Best-of-K** — keep only the top-scoring passing response per prompt. Less data, higher quality.
- **All-passing** — keep every passing response. More data, some redundancy.

The Llama-3 paper uses best-of-K for most stages. DeepSeek-R1 keeps **multiple** passing responses per prompt in its Stage 3 resample to get 600k reasoning samples from a smaller set of prompts.

### Self-bootstrapping loop

Rejection sampling becomes a **self-improvement loop** when you iterate:

```
model_0 = base or SFT
for round in 1..N:
    data_round = rejection_sample(model_{round-1}, K, filter)
    model_round = SFT(model_{round-1}, data_round)
```

Each round, the generator is stronger than the last, so the *passing* responses are a stronger dataset, so the next SFT produces a stronger model. The ceiling is set by the filter, not by the starting model — if the filter is a rule verifier on math, the model can eventually reach whatever the verifier defines as "correct" for every training prompt.

---

## Three canonical uses

### 1. Llama 3 post-training

Llama 3 uses rejection sampling as the dominant SFT-data-generation method at multiple stages. For each prompt they sample `K` completions with the current best model (earlier-iteration Llama), apply a reward model (and/or rule-based checks for math/code), keep the best response, and SFT on it. Several rounds; the data grows with the model.

### 2. DeepSeek-R1 Stage 3 — the "big resample"

The pivotal stage in the R1 pipeline. After reasoning-oriented RL produces a checkpoint with strong math/code capability (but messy traces), DeepSeek:

- **Samples many rollouts per prompt** from the post-Stage-2 checkpoint.
- **For verifiable prompts** (math, code with tests): keeps only correct ones via rule verifier.
- **For non-verifiable prompts** (open-ended reasoning, science QA): uses **DeepSeek-V3 as a generative judge**, comparing the response to a ground-truth answer — a specific, cheap form of LLM-as-judge.
- **Filters further** by dropping mixed-language CoT, overlong paragraphs, stray code blocks, and unreadable output.
- Result: **~600k reasoning-related samples**. Then **~200k non-reasoning samples** (writing, QA, translation, etc.) are added — reused from the DeepSeek-V3 SFT pipeline, with some augmented by a synthetic short CoT.
- Total ~**800k samples, 2 epochs** of full SFT over DeepSeek-V3-Base. This becomes the checkpoint that Stage 4 alignment RL runs on top of, and this is also the dataset used to **distill** R1 into smaller dense models (Qwen2.5, Llama 3 series).

The key structural pattern: **use RL to produce capability, use rejection sampling to produce legibility and coverage.** RL gets you a model that can solve the problems; rejection sampling gets you a dataset where every example is a *clean, correct* demonstration.

### 3. Kimi k1.5's shortest-rejection-sampling for long2short

A length-tiebreak variant: sample `n = 8` rollouts per prompt from a long-CoT model, filter to correct responses, keep the **shortest correct**, SFT on the result. Specifically targeted at compressing long-CoT capability into a short-CoT model — the filter is "correct", the tiebreak is `len(o)`. One of four methods Kimi compares for [long2short](reasoning/long2short.md); middle-of-the-pack quality, simplest to implement.

### 4. Synthetic data pipelines

Many synthetic-data pipelines (Orca, Phi, WizardMath, a dozen open recipes) are essentially rejection sampling with a judge model. A teacher generates candidate responses; a judge (often GPT-4 or Claude) scores them; the top-scoring set is used for SFT. Same structure, different roles.

---

## Why it matters

- **Cheap way to scale high-quality SFT data.** Much cheaper than human labels; aligned with whatever filter you use.
- **The canonical "capability → data" move.** A capable model can label its own training data — *given a filter that is more reliable than the model's average behavior*. This is how a lot of post-training compute gets spent.
- **Enables RL-without-distillation-at-deploy.** A small dense model distilled from rejection-sampled R1 traces (via SFT) outperforms that same small model trained with RL from scratch. Rejection sampling is the bridge between RL capability and deployable dense checkpoints. See the R1 distillation result in [deepseek-r1](../case-studies/deepseek-r1.md).
- **Composes with RL.** Rejection-sampling SFT → RL is the Tülu-3 / R1 / Llama-3 pattern. Each step uses a different signal: SFT teaches format on clean data; RL pushes the capability ceiling.

---

## Gotchas & tricks

- **Yield collapses on too-hard prompts.** If the generator can't solve a prompt with any of `K` rollouts, the prompt is dropped — the harder the training distribution, the lower the yield. Mitigation: curriculum (easier prompts early), increase `K`, use a stronger generator for data generation than for deployment.
- **Filter bias shapes the model.** The SFT model reproduces whatever the filter privileges. If your rule verifier only accepts exact string match on the final answer, the SFT model learns to emit bare numbers and skimps on explanation. Design the filter as carefully as the dataset.
- **Don't forget diversity filters.** Best-of-K on pure reward leads to mode collapse in the dataset (same phrasings, same structure). Llama 3 and R1 both apply secondary diversity filters (dedup, length variance, template detection).
- **LLM-as-judge is leaky.** Generative judges have idiosyncratic preferences. Ensemble with rules where possible.
- **`K` has diminishing returns.** Doubling `K` from 8 to 16 is usually worth it; from 32 to 64 is usually not. Exception: very hard prompts where low single-try pass rate makes large `K` the only option.
- **Multi-round self-bootstrapping can drift.** Each round's generator is trained on its *predecessor's* passing outputs; style and failure modes compound. Keep a stable, human-curated eval and watch for narrowing.
- **Not the same as "rejection sampling" in statistics.** Classical rejection sampling is a generic technique for sampling from a target distribution by proposing from a simpler one and accepting with probability proportional to a ratio. The SFT-data-gen usage reuses the name (propose responses, accept only some) but is rarely framed in the density-ratio sense.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — describes rejection-sampling SFT as a primary post-training data-generation method.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — uses rejection sampling in Stage 3 (~600k reasoning + ~200k non-reasoning = ~800k SFT samples) off the Stage-2 RL checkpoint, driven by rule verifier + DeepSeek-V3 generative judge.
- Paper: *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* — AI2, 2024 — canonical open recipe, uses rejection sampling on SFT-completion stages.
- Paper: *Training Verifiers to Solve Math Word Problems* — Cobbe et al., 2021 — early example of using a learned verifier to rerank/filter model outputs on GSM8K (see [orm](reasoning/orm.md)).
- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI, 2025 — uses shortest-rejection-sampling (`n=8`, length-tiebreak) as one of four [long2short](reasoning/long2short.md) compression methods.
