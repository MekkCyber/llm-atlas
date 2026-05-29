# llm-atlas knowledge graph — structure cheat sheet

The repo is a hand-curated wiki organized as **one concept per file**, with cross-links via `Prereqs:` / `Related:` lines at the top of each file. This cheat sheet exists so the digest skill can map new papers to the right places in the graph.

## Two kinds of concept files

Both live inside topical folders, distinguished by a one-line italic label under the title.

- **Depth file** — `<technique>.md` (kebab-case). One specific technique grounded in its source paper(s). Template: `TEMPLATE-DEPTH.md`.
  - Sections: TL;DR · Prereqs/Related · What it is · How it works · Why it matters · Gotchas & tricks · Sources.
- **Taxonomy file** — `_<category>.md` (leading underscore). Overview of a class of techniques, links to the depth files it covers. Template: `TEMPLATE-TAXONOMY.md`.
  - Sections: TL;DR · Related taxonomies · Depth files covered · The problem · Shared pattern · Variants (table) · How to choose · Adjacent but distinct · Sources.

**Rule of thumb when suggesting:**
- A paper introduces a *single technique*? → suggest a **depth file** in the matching folder.
- A paper surveys / contrasts several variants? → suggest (or update) a **taxonomy file**.
- A paper is a *milestone end-to-end system* (a tech report, frontier model, big release)? → suggest a **case study** in `case-studies/`.

## Folder map

| Folder | Holds |
| --- | --- |
| `fundamentals/` | Tokenization, embeddings, positional encoding, optimizers, losses, normalization, activations, initialization, prompting. |
| `architectures/` | Transformers, SSM (Mamba, S4), MoE, attention variants (MLA, GQA), positional encodings. |
| `pre-training/` | Data pipelines, parallelism (FSDP/TP/PP), mixed precision, scaling laws, training stability. |
| `post-training/` | RLHF, PPO, GRPO, DPO, reward modeling, preference learning, rollouts. |
| `post-training/fine-tuning/` | SFT variants, LoRA, QLoRA, instruction tuning. |
| `post-training/reasoning/` | Long-CoT RL, process rewards, reasoning-specific training. |
| `systems/` | Ray, distributed scheduling, fault tolerance, checkpointing, orchestration, rollout workers. |
| `inference/` | KV cache, paged attention, continuous batching, speculative decoding, vLLM/SGLang, prefill/decode disaggregation. |
| `quantization/` | FP8, INT4, MXFP4, GPTQ, AWQ, calibration, mixed-precision tradeoffs. |
| `data/` | Curation, filtering, deduplication, tokenization strategy, mixtures, synthetic data. |
| `multimodal/` | Vision encoders, projection layers, multimodal pretraining/post-training, video. |
| `agents/` | Tool calling, function schemas, multi-turn execution, MCP, training with environments. |
| `evaluation/` | Benchmarks, reward model evaluation, pairwise comparison, contamination. |
| `safety/` | RLHF for safety, constitutional AI, red-teaming, jailbreaks, refusal training, dangerous capability evals, scheming. |
| `interpretability/` | Probing, mech-interp, circuits, SAEs, activation steering, logit lens. |
| `case-studies/` | End-to-end systems and milestone papers — `composer2.md`, `deepseek-r1.md`, `deepseek-v3.md`, `kimi-k1-5.md`, `olmo-2.md`, `qwen2-5.md`, etc. |

## Cross-file link format

Every concept file starts with (after TL;DR):

```
**Prereqs:** [link](../path/to.md), [link](../path/to.md)
**Related:** [link](../path/to.md), [link](../path/to.md)
```

Case studies use a **Related concepts:** line with `·`-separated links instead. See `case-studies/deepseek-v3.md` for the canonical example.

Paths from a concept file are **relative**: `../architectures/mla.md` from `case-studies/`, `../post-training/grpo.md` from `architectures/`. **From a digest file at `daily-papers/<date>.md`, all KG links should be `../<folder>/<file>.md`.**

## Naming conventions

- Depth files: technique-name in kebab-case. Examples: `rope.md`, `gqa.md`, `adamw.md`, `mla.md`, `dualpipe.md`, `fp8-training.md`, `grpo.md`.
- Taxonomy files: `_<category>.md`. Examples: `_positional-encoding.md`, `_optimizers.md`, `_attention-variants.md`, `_moe.md`, `_jailbreaks.md`, `_rl.md`, `_rewards.md`.
- Case studies: kebab-case model/system name. Examples: `deepseek-v3.md`, `kimi-k1-5.md`, `olmo-2.md`, `qwen2-5.md`.
- Always lowercase.

## Choosing depth vs case study (when suggesting)

| Looks like | Suggest |
| --- | --- |
| A new training technique (loss, scheduler, parallelism trick, RL algo variant) | depth file in matching folder |
| A new attention/MoE/normalization variant | depth file in `architectures/` |
| A survey / contrast of several variants of the same class | taxonomy file (or update to an existing `_<cat>.md`) |
| A frontier model tech report (Llama X, DeepSeek X, Qwen X, Kimi X) | case study + a depth file per individually-novel innovation |
| A safety / interp paper introducing a new evaluation or attack | depth file in `safety/` or `evaluation/` or `interpretability/` |
| An agentic system using existing techniques | case study only |

## Attribution rule (matters for "suggested new pages")

A depth page belongs to the paper that *triggered its creation*. Many depth pages cite earlier primary sources in their `Sources` section — those are references, not reads. When suggesting a new depth page, prefer attaching it to the **earliest primary source** if known; otherwise the current paper.
