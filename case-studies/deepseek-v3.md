# Case Study: DeepSeek-V3

*A 671B-total / 37B-active MoE that reached GPT-4o / Claude 3.5 Sonnet benchmark parity for ~$5.6M of training compute. The interesting part is not the final weights — it's that the paper is a dense bag of systems and algorithmic innovations, any one of which would be a real contribution on its own.*

**Related concepts:** [mla](../architectures/mla.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [sequence-wise-balance-loss](../architectures/sequence-wise-balance-loss.md) · [load-balancing-loss](../architectures/load-balancing-loss.md) · [capacity-factor](../architectures/capacity-factor.md) · [mtp](../pre-training/mtp.md) · [fp8-training](../pre-training/fp8-training.md) · [dualpipe](../systems/dualpipe.md) · [grpo](../post-training/grpo.md) · [rlvr](../post-training/rlvr.md)

---

## What this is

DeepSeek-V3, released December 2024 by DeepSeek AI. A mixture-of-experts decoder transformer: 671B total parameters, 37B activated per token, trained from scratch on 14.8T tokens in 2.788M H800 GPU-hours (~$5.576M at $2/GPU-hour). At release it was the best open-source model on most benchmarks and roughly on par with GPT-4o and Claude 3.5 Sonnet.

The paper matters because of what it bundles together: **MLA** (KV-cache compression), **DeepSeekMoE** (fine-grained + shared experts), **aux-loss-free balancing**, **MTP** (multi-token prediction auxiliary objective), end-to-end **FP8 training** with fine-grained quantization, **DualPipe** pipeline parallelism, custom **all-to-all** kernels, and a post-training pipeline that distills long-CoT reasoning from DeepSeek-R1 and then applies **GRPO** RL. Any one of these alone would be a paper; V3 combines them and validates the stack at frontier scale.

---

## Architecture at a glance

```
61 transformer layers
  ├─ first 3:   attention (MLA) + dense FFN
  └─ layers 4–61: attention (MLA) + MoE FFN (DeepSeekMoE + aux-loss-free)

d_model       = 7168
heads         = 128,  d_head = 128
MLA latents   = d_c (KV) = 512,  d_c' (Q) = 1536,  d_h^R (RoPE slice) = 64
MoE per layer = 1 shared expert + 256 routed experts
              top-8 routed + 1 shared activated per token
              expert intermediate dim = 2048 (small — fine-grained)
              node-limited routing: M = 4 nodes max per token

+ Multi-Token Prediction head (depth-1)
  shares embedding and LM head with main model
```

All of this is covered in the concept pages — see the **Related concepts** line above.

---

## Training infrastructure

### Parallelism

```
tensor parallelism:      none at training time
pipeline parallelism:    16-way, DualPipe scheduling
expert parallelism:      64-way (over 8 nodes)
data parallelism:        ZeRO-1
```

No tensor parallelism in training is unusual for a model this size. It's enabled by DualPipe keeping pipeline utilization high and by DeepEP's custom MoE all-to-all kernels. TP is used only at inference time (TP=4 during prefill/decode).

### FP8 everywhere that's safe

E4M3 throughout, 1×128 activation tiles, 128×128 weight blocks, FP32 accumulation every 128 inner-dim elements. Embeddings, LM head, MoE gating, norms, and attention stay in BF16/FP32. Master weights FP32, AdamW `m, v` in BF16. See [fp8-training](../pre-training/fp8-training.md).

### DualPipe + 20 communication SMs per GPU

Two pipelines flowing in opposite directions, forward/backward compute and all-to-all communication scheduled to overlap. 20 of 132 H800 SMs per GPU reserved for comm with warp specialization. Doubles per-GPU parameter memory, eliminates bubbles. See [dualpipe](../systems/dualpipe.md).

### H800 constraints shape the design

H800 is the export-controlled variant of H100 — same compute, reduced NVLink (400 GB/s vs 900 GB/s) and IB bandwidth. DualPipe, node-limited routing (M=4), custom all-to-all kernels, and the 20-SM comm reservation all exist to work around this interconnect ceiling. On unrestricted H100 clusters some of the engineering is over-solution.

---

## Training recipe

### Pre-training

- **14.8T tokens.** Higher math/code ratio than DeepSeek-V2.
- **128K-token byte-level BPE vocab.**
- **AdamW.** β1 = 0.9, β2 = 0.95, weight decay = 0.1.
- **LR schedule** (not a simple cosine):
    - Linear warmup 0 → 2.2e-4 over 2 000 steps
    - Constant at 2.2e-4 until 10T tokens consumed
    - Cosine decay 2.2e-4 → 2.2e-5 over the next 4.3T
    - Step-constant tail: 2.2e-5 for 333B tokens, then 7.3e-6 for final 167B
- **Aux-loss-free bias update step** `γ = 0.001` for first 14.3T tokens, `γ = 0` for final 500B (biases frozen).
- **Complementary sequence-wise balance loss** with coefficient `α = 1e-4`.
- **MTP coefficient** `λ = 0.3` for first 10T tokens, `λ = 0.1` for final 4.8T.

### Context extension (YaRN)

Two 1 000-step post-pretraining stages:
1. **4K → 32K**: batch 1920, seq 32K.
2. **32K → 128K**: batch 480, seq 128K.

Both stages use LR 7.3e-6. YaRN parameters: scale s=40, α=1, β=32.

### Post-training

**SFT** on 1.5M instances spanning reasoning (math, code, logic), general (writing, QA), role-play, and safety. Reasoning-task data is **distilled from DeepSeek-R1** — for each reasoning prompt they sample long chain-of-thought traces from an internal R1 model and use them as SFT targets. V3 inherits R1's reasoning patterns without running long-CoT RL itself.

**RL** via [GRPO](../post-training/grpo.md) with two reward sources:
- **Rule-based** for verifiable tasks (math answer match, unit-test pass, format checks) — this is [RLVR](../post-training/rlvr.md).
- **Model-based** for open-ended tasks — a preference-tuned reward model.

---

## Training cost (Table 1 of the paper)

| Phase | H800 GPU-hours |
|---|---|
| Pre-training | 2 664 K |
| Context extension (32K + 128K) | 119 K |
| Post-training | 5 K |
| **Total** | **2 788 K** (~$5.576M at $2/GPU-hour) |

This number is the most-cited figure from the paper. It's 5–10× cheaper than frontier-scale training runs of the same era, and the concrete demonstration that FP8 + DualPipe + fine-grained MoE can shave frontier-scale costs by large factors.

---

## Evaluation snapshot

Selected benchmarks from the paper's eval tables; DeepSeek-V3 refers to the chat model unless noted.

| Benchmark | DeepSeek-V3 | GPT-4o | Claude 3.5 Sonnet |
|---|---|---|---|
| MMLU (EM) | 88.5 | 87.2 | 88.3 |
| MMLU-Pro (EM) | 75.9 | 72.6 | 78.0 |
| GPQA-Diamond (Pass@1) | 59.1 | 49.9 | 65.0 |
| MATH-500 (EM) | 90.2 | 74.6 | 78.3 |
| **AIME 2024 (Pass@1)** | **39.2** | 9.3 | 16.0 |
| HumanEval-Mul (Pass@1) | 82.6 | 80.5 | 81.7 |
| LiveCodeBench (Pass@1-COT) | 40.5 | 33.4 | 36.3 |

The standout is math — AIME 2024 at 39.2 vs GPT-4o's 9.3 — driven by the R1-distilled SFT data.

---

## Inference mode: MTP for speculative decoding

DeepSeek-V3 ships with the MTP module retained at inference. The module drafts token `t+2` from the main model's hidden state at position `t` plus the embedding of `x_{t+1}`. The main model verifies: if its own prediction at `t+2` matches the draft, accept both; else re-decode.

Reported acceptance rate: **85–90%**. Effective speedup: **~1.8× tokens per second** on DeepSeek's own serving stack.

---

## Key takeaways

1. **Frontier FP8 training is a solved problem.** Fine-grained per-tile scaling + FP32-promoted accumulation + selective high-precision components are the recipe. Details in [fp8-training](../pre-training/fp8-training.md). Every new frontier run from 2025 onward should evaluate this stack.

2. **Auxiliary-loss-free MoE balancing is strictly better.** The per-expert bias control loop (with no gradient) outperforms classical aux-loss at frontier scale, and costs ~nothing. Use this as the default. See [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md).

3. **Fine-grained experts + shared expert is the MoE frontier.** 256 routed + 1 shared, top-8 routing at 37B active is a new Pareto point. See [deepseek-moe](../architectures/deepseek-moe.md).

4. **MLA removes the KV-cache bottleneck for long context.** 5–10× smaller cache vs standard MHA at matched quality. See [mla](../architectures/mla.md).

5. **MTP gives both a training-time quality lift and an ~1.8× inference speedup** via self-speculative decoding. See [mtp](../pre-training/mtp.md).

6. **DualPipe + 20-SM comm kernels make the interconnect disappear.** Pipeline bubbles and all-to-all cost overlap with compute. See [dualpipe](../systems/dualpipe.md). H800 constraints drove the design; H100 clusters benefit somewhat less.

7. **R1 distillation shortcuts long-CoT RL.** You don't need to do reasoning RL on your production model if you have a reasoning-specialized sibling (R1) whose traces you can distill via SFT.

8. **Training cost is a systems outcome, not a compute quota.** $5.576M for a GPT-4o-class model is downstream of a dozen individually-significant efficiency wins stacked together, not a single breakthrough.

---

## What's still opaque

- **No intermediate checkpoints released.** Compare to OLMo 2's fully open trajectory. V3 is open weights, not open training.
- **FP8 training code** was not open at paper release (DeepEP / comm kernels were released; the full FP8 framework was documented but not shipped as a drop-in framework).
- **Tokenizer composition** is not fully detailed — 128K byte-level BPE but the corpus it was trained on is under-specified.
- **Safety / refusal training** is mentioned but not characterized quantitatively.

---

*Pairs well with:* the [OLMo 2 case study](olmo-2.md) for contrast — OLMo 2 is the reproducibility reference, V3 is the frontier-efficiency reference. Different virtues, complementary reads.
