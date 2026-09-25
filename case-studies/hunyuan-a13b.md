# Case Study: Hunyuan-A13B

*An open-source 80B-total / 13B-active MoE with a **dual-mode Chain-of-Thought** framework — one model that dispatches routine queries through a "fast thinking" path and complex problems through a "slow thinking" long-CoT path, trained with a two-stage RL recipe over both modes.*

**Related concepts:** [_moe](../architectures/_moe.md) · [multi-head-attention](../architectures/multi-head-attention.md) · [grpo](../post-training/grpo.md) · [rlvr](../post-training/rlvr.md) · [long-cot-rl](../post-training/reasoning/long-cot-rl.md) · [long2short](../post-training/reasoning/long2short.md) · [orm](../post-training/reasoning/orm.md) · [dual-mode-cot](../post-training/reasoning/dual-mode-cot.md) · [rope](../fundamentals/rope.md)

---

## What this is

Hunyuan-A13B, released September 2026 by Tencent's Hunyuan team. A mixture-of-experts decoder transformer: **80B total parameters, 13B activated per token**, pretrained on **20T rigorously-filtered tokens** with heavy STEM curation, then aligned via SFT and large-scale reinforcement learning.

The distinguishing feature is the **dual-mode Chain-of-Thought** framework — a single set of weights supports both fast, short-form thinking and slow, long-CoT reasoning, with the mode selectable at prompt time (and internally on complexity signals). This targets the "one model, two latency budgets" niche where production traffic mixes routine chatty queries with occasional hard multi-step problems.

Positioned between DeepSeek-V3 (671B/37B) and Qwen3-A22B on the open-source MoE Pareto curve, with a much smaller total-parameter footprint but a similar active-parameter target.

---

## Architecture at a glance

```
32 transformer layers
  ├─ attention: GQA — 32 Q heads / 8 KV heads
  └─ MoE FFN:  1 shared expert + 64 fine-grained non-shared experts
              top-8 routed + 1 shared activated per token

d_model              = 4096
d_ffn (per expert)   = 3072   (fine-grained — small experts)
activation           = SwiGLU
positional encoding  = NTK-aware RoPE  (α = 50 at 32K, α = 1000 at 256K)
vocabulary           = 128K (same tokenizer as Hunyuan-Large)
```

Architectural choices are all "known-good" primitives assembled cleanly:
- **GQA** (8 KV heads for 32 Q heads) — 4× KV cache reduction vs standard MHA.
- **Fine-grained MoE with shared expert** — 1 always-on + 64 routed follows the [DeepSeekMoE](../architectures/README.md) template.
- **NTK-aware RoPE** — different α at different context stages during training rather than a single YaRN-style extension pass.
- **SwiGLU** — standard.

No new architecture invention; the interesting work is in the training recipe.

---

## Training recipe

### Pretraining — four context stages

| Stage | Context | Purpose | Notes |
| --- | --- | --- | --- |
| Foundation | 4 096 | Base LM training | Bulk of the 20T tokens |
| Annealing | 8 192 | Data-mixture adjust | Higher STEM fraction |
| Long-context I | 32 768 | Context extension | NTK α = 50 |
| Long-context II | 262 144 | Long-context extension | NTK α = 1 000 |

**Data.** 20T tokens total, with a curated **250B-token high-quality STEM pretraining corpus** injected at the annealing stage to strengthen math, science, and reasoning surface. Same tokenizer as Hunyuan-Large, 128K vocab.

**LR schedule.** Warmup to peak 3×10⁻⁴, cosine decay to 3×10⁻⁵ over 13.5T tokens, then a **fast-annealing tail** to 8×10⁻⁶ over the final 300B tokens. The tail is the typical [wsd-schedule](../pre-training/wsd-schedule.md)-adjacent trick — reserves a large fraction of the LR range for late-training data mixture changes without a full re-warmup.

**Optimizer.** AdamW (standard; explicit hyperparameters not stated in the abstract page).

### Post-training

**SFT** on high-quality instruction data covering reasoning (math, coding, logic, science), creative writing, multilingual tasks, knowledge QA, and agent scenarios. Volume not explicitly quantified.

**Reinforcement learning** in two stages:

1. **Reasoning-RL stage.** [GRPO](../post-training/grpo.md) with **outcome reward models** and **sandbox execution feedback**. 150K reasoning prompts. This is the [long-CoT RL](../post-training/reasoning/long-cot-rl.md) stage — trains the slow-thinking path.
2. **General-RL stage.** Generative reward models comparing rollouts against reference answers, plus domain-specific reward services. Aligns the fast-thinking path and open-ended generation.

The two stages are the mechanism by which the same weights learn both fast and slow behaviors — see [dual-mode-cot](../post-training/reasoning/dual-mode-cot.md) for the training-side details.

---

## The dual-mode CoT framework

The novel contribution. In a single model, expose two reasoning regimes chosen at prompt time:

- **Fast thinking** — short-answer path, low-latency, minimal CoT tokens. Trained on the general-RL stage.
- **Slow thinking** — long-CoT path, with the reasoning-RL rewards allowed to lengthen traces as needed for hard problems.

Because both modes share the same weights, dispatch is a *conditioning* mechanism (prompt tag, system message, or an internal complexity classifier) rather than a router across models. Contrasts with:

- **DeepSeek-V3 + R1** — two separate models; V3 distills R1's traces via SFT and doesn't run long-CoT RL itself.
- **Qwen3** — reasoning mode is a system flag but the depth control is coarser and less integrated with training-time reward shaping.

See [dual-mode-cot](../post-training/reasoning/dual-mode-cot.md) for the training-side mechanism.

---

## Evaluation snapshot

Selected numbers from the tech report:

| Benchmark | Score | Mode |
| --- | --- | --- |
| MATH | 85.4 | fast-thinking |
| AIME 2024 | 87.3 | slow-thinking |
| GSM8K | 91.83 | — |
| MMLU | 88.17 | — |
| EvalPlus (coding) | 78.64 | — |

The AIME 2024 slow-thinking score is the marquee number: **87.3** competitive with much larger dense frontier models. The MATH gap between fast (72.35 base → 85.4 fast-thinking) illustrates the intended tradeoff — fast thinking is not a hobbled mode, but slow thinking wins on the hardest problems.

---

## Key takeaways

1. **Dual-mode CoT is a training-time answer to adaptive reasoning depth.** Most production systems currently emulate this with an external router across two models or a hard length cap. Hunyuan-A13B compresses both into one set of weights, trained via a two-stage RL that shapes each mode. See [dual-mode-cot](../post-training/reasoning/dual-mode-cot.md).

2. **13B active parameters is a viable frontier-adjacent operating point.** With GQA, fine-grained MoE, and strong post-training, a 13B-active model can approach dense models several times its size on math and reasoning — as long as the training data (250B curated STEM) and RL are aggressive.

3. **STEM data curation is a first-class training-recipe lever.** Segregating a 250B-token high-quality STEM slice and injecting it at the annealing stage is a specific, replicable move — not just "we filtered our data."

4. **Two-stage RL beats single-stage.** Splitting reasoning-RL and general-RL lets each stage optimize its own reward signal without cross-contamination — reasoning gets outcome-based sandbox feedback; general gets preference-style generative rewards.

5. **NTK-aware RoPE with staged α values is competitive with YaRN.** Different α at different context stages, rather than a single YaRN extension pass, reaches 256K without a dedicated context-extension recipe.

---

## What's still opaque

- **Parallelism and infrastructure.** The abstract page does not describe the training-cluster setup, parallelism strategy, or exact optimizer hyperparameters. Presumably detailed in the full report.
- **Distinction between "fast" and "slow" thinking at inference.** Whether dispatch is fully user-controlled, learned as an internal classifier, or a hybrid is not fully spelled out.
- **Reward model architecture.** Generative reward models are used in the general-RL stage; the depth of these RMs and their training data is not detailed here.
- **Safety training.** Mentioned in passing; not quantitatively characterized on this page.

---

*Pairs well with:* the [DeepSeek-V3 case study](deepseek-v3.md) for a larger-scale MoE reference, and [Kimi k1.5](kimi-k1-5.md) for a different take on long-CoT RL. Hunyuan-A13B is the "small-active, dual-mode CoT" corner of the same design space.
