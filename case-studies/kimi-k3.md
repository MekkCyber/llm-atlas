# Case Study: Kimi K3

*Moonshot AI's third-generation frontier model, released July 2026. A **2.8T-parameter Mixture-of-Experts** with **104B activated parameters**, **native vision**, and a **1M-token context window**, positioned as a fully-open competitor to closed frontier systems. The lineage continues the Kimi K2 (open) and Kimi k1.5 (long-CoT reasoning) recipes; the story is scaling MoE another turn while shipping vision and 1M context in the same weights.*

**Related concepts:** [_moe](../architectures/_moe.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [mla](../architectures/mla.md) · [mtp](../pre-training/mtp.md) · [fp8-training](../pre-training/fp8-training.md) · [fp8](../quantization/fp8.md) · [dualpipe](../systems/dualpipe.md) · [grpo](../post-training/grpo.md) · [rlvr](../post-training/rlvr.md) · [long-cot-rl](../post-training/reasoning/long-cot-rl.md) · [online-policy-mirror-descent](../post-training/reasoning/online-policy-mirror-descent.md) · [long2short](../post-training/reasoning/long2short.md) · [partial-rollouts](../systems/partial-rollouts.md) · [kimi-k1-5 case study](kimi-k1-5.md)

---

## What this is

**Kimi K3**, released July 2026 by Moonshot AI. arXiv 2607.24653. A decoder-only Mixture-of-Experts language + vision model with a **2.8T total-parameter / 104B activated-parameter** MoE, **native (non-projection) vision input**, and a **1M-token context window**, released as open weights.

Positioning:

- **Fully open at frontier scale.** K3 is the first open release above the ~100B-active MoE frontier that ships with native vision and 1M context in the same weights. Where DeepSeek-V3 ran at 671B/37B and OLMo 2 established the reproducibility reference at ~13B–70B dense, K3 pushes both total (2.8T) and active (104B) budgets higher and adds vision natively.
- **Higher active budget, sparser MoE.** The 104B / 2.8T = ~3.7% activation ratio is *sparser* than V3's ~5.5% (37B / 671B) while activating more parameters per token in absolute terms — a Pareto point that keeps reasoning-heavy behavior at frontier quality without paying dense-104B compute.
- **Continues the Kimi long-CoT-RL playbook** started in k1.5 (online policy mirror descent, length-penalty rewards, partial rollouts) and extended to agentic tool-use trajectories.

Direct inheritance from prior Kimi releases (k1.5, K2) is discussed below; details specific to K3 that were introduced in this tech report are called out inline.

---

## Architecture at a glance (headline facts only)

The exact per-layer breakdown is not in the abstract; the following headline numbers are stated:

```
2.8T total parameters (MoE)
104B activated per token
Native vision — image tokens flow into the same backbone as text (no projection stage)
1M-token context window
Fully open weights
```

Plausibly inherited from the deepseek-v3 / K2 frontier-MoE recipe (to be confirmed against the full report):
- **MLA** or an MLA-family attention variant for KV-cache compression, needed to keep 1M-context memory tractable.
- **Fine-grained MoE** with a small shared expert bank plus many small routed experts (see [deepseek-moe](../architectures/deepseek-moe.md)).
- **Aux-loss-free balancing** — the current default for frontier-scale MoE (see [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md)).
- **FP8 training** with fine-grained per-tile scaling (see [fp8-training](../pre-training/fp8-training.md)) — required at this parameter count on any realistic H-series cluster.
- **DualPipe-style pipeline scheduling + custom all-to-all MoE communication** (see [dualpipe](../systems/dualpipe.md)) — the standard frontier-MoE systems recipe.

These are marked as inheritance rather than K3-specific novelty; the abstract does not commit to any of them, and the full breakdown should be filled in from the tech report body when it's read.

---

## Vision, native and in-backbone

The abstract highlights **native vision capabilities** as a first-class feature. In the Kimi lineage this contrasts with the projection-layer pattern used in early VLMs (a separate vision encoder → a projector → the LLM). The k1.5 report already introduced language-first-then-multimodal progressive pretraining with a vision tower that is unfrozen partway through; K3 continues this direction, with visual tokens flowing directly into the shared backbone rather than through a bridge module.

Consequences (to verify against the full report):
- One shared context window for text and image tokens — the 1M budget is genuinely multimodal.
- A single tokenizer/decoder stack — no cross-modal alignment layer separate from the backbone.
- Vision RL post-training folds into the same policy-optimization loop as text (a k1.5 hallmark; expected to carry forward).

---

## The 1M-token context window

1M tokens is the headline capability jump over K2. At this length, KV-cache footprint dominates memory and standard attention becomes untenable:

- **KV-cache compression** via MLA (or a family variant) is effectively required. See [mla](../architectures/mla.md).
- **Long-context activation stage** — the k1.5 recipe extended context in three stages (4K → 32K → 128K) with progressively adapted RoPE base. K3 continues this pattern with an additional stage into the 1M regime. Exact numbers await the full report.
- **Inference infra changes** — 1M-context serving benefits from prefill/decode disaggregation and long-context-aware attention kernels; not detailed in the abstract.

---

## Post-training pipeline

The abstract does not enumerate stages, but the K3 report is stated to extend the Kimi k1.5 long-CoT-RL playbook to *agentic tool-use trajectories*. The expected pipeline (from k1.5 continuity, to be verified):

```
Pretraining
  ├─ language-first + progressive multimodal
  ├─ cooldown with synthetic rejection-sampled QA
  └─ long-context activation up to 1M

Vanilla SFT
  └─ text + text-vision mixtures at successive context lengths

Long-CoT SFT cold-start
  └─ planning / reflection / exploration traces

RL (long-CoT + agentic)
  ├─ online policy mirror descent (or GRPO-family; see grpo)
  ├─ length-penalty rewards to fight overthinking
  ├─ partial rollouts for long-context RL efficiency
  ├─ rule-based rewards for verifiable tasks (RLVR)
  └─ agentic-trajectory RL over tool-use environments

long2short
  └─ compression of long-CoT capability into short-CoT
```

The novel K3 addition on top of this k1.5 shape is agentic RL — training the model in tool-use environments so long-horizon planning and function calling are covered under the same RL loop as reasoning. Details await the full report.

---

## Evaluation framing (to fill in from full report)

The abstract advertises frontier-level performance on math, code, and multimodal benchmarks with the explicit framing that a fully-open MoE at this scale matches closed systems. Specific numbers on AIME, MATH-500, LiveCodeBench, MMLU, GPQA-Diamond, agentic benchmarks (SWE-bench, WebArena), and multimodal benchmarks (MMMU, MathVista) are not in the abstract; the case-study evaluation table should be filled in from the tech report body.

Categories to look for when reading the full report:
- **Math reasoning** — AIME 2026, MATH-500, HMMT
- **Code reasoning** — LiveCodeBench, HumanEval-Mul, SWE-bench Verified
- **Knowledge** — MMLU-Pro, GPQA-Diamond
- **Long-context** — RULER, LongBench, needle-in-a-haystack at 1M
- **Multimodal** — MMMU, MathVista, ChartQA, InfoVQA
- **Agentic** — SWE-bench, WebArena, GAIA

---

## What makes K3 significant

1. **First open MoE past ~100B active with native vision + 1M context in one release.** Frontier open weights up to now (V3, K2, Llama variants) picked at most two of {large MoE, native vision, ≥1M context}. K3 ships all three.
2. **The 104B / 2.8T sparsity ratio is a new Pareto point.** Higher absolute active budget than V3, but sparser — evidence that expert count scales further than 256 usefully when routing and balancing are handled properly.
3. **Serveable via community MoE stacks.** Not a demonstration weight release — designed to run on open MoE inference infrastructure, so the community can actually consume the model.
4. **Forces closed labs to keep the open-vs-closed margin narrow.** Similar strategic effect to V3 in December 2024, but at a higher active-parameter tier and with vision-native scope.
5. **Continues the Kimi long-CoT-RL lineage as the frontier reasoning recipe.** OPMD + length penalty + partial rollouts move up from k1.5's context regime to 1M-native and gain an agentic-trajectory arm.

---

## What's opaque from the abstract

The case study should be updated once the full tech report is available. Specifically:

- **Per-layer architecture** — number of layers, attention variant used (MLA / MLA-variant / other), routed and shared expert counts, expert intermediate dim.
- **MoE routing** — top-K per token, node-limited routing thresholds, load-balancing strategy.
- **Training data** — corpus composition, tokens consumed, multimodal fraction.
- **Training precision and systems** — FP8 vs BF16, parallelism strategy, DualPipe-style scheduling.
- **Long-context extension** — YaRN or alternative, stage schedule out to 1M.
- **Post-training** — full pipeline, RL algorithm, reward shaping specifics, agentic-RL environment(s).
- **Evaluation numbers** — the benchmark table.
- **Compute cost** — total GPU-hours, dollar figure.
- **Vision tower** — encoder architecture, unfreezing schedule, multimodal token budget per image.

Every one of these is called out in the standard llm-atlas case-study format (see [deepseek-v3](deepseek-v3.md) and [kimi-k1-5](kimi-k1-5.md)) and should be filled in from the body of arXiv 2607.24653.

---

## Sources

- Paper: *Kimi K3: Open Frontier Intelligence* — Kimi Team (Moonshot AI), 2026 — [arXiv:2607.24653](https://arxiv.org/abs/2607.24653)
- Predecessor: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — [kimi-k1-5 case study](kimi-k1-5.md)
- Comparison point: *DeepSeek-V3 Technical Report* — [deepseek-v3 case study](deepseek-v3.md)

*Pairs well with:* [deepseek-v3](deepseek-v3.md) — same recipe genus (large-sparse MoE, FP8, long-context, MTP, GRPO-family RL) at a smaller total but comparable active-parameter tier. Reading K3 alongside V3 shows the two dominant frontier-MoE trajectories converging: keep sparsening the routing, add vision natively, stretch context.
