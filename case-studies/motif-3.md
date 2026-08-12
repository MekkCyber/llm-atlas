# Case Study: Motif 3

*A 314B-total / 13.2B-active MoE tech report that bundles a new attention variant (GDLA), fine-grained routing at 384 experts, MXFP8 training, MTP, and a distillation-based post-training pipeline (Multi-teacher On-Policy Distillation). Positions itself alongside DeepSeek-V3 as the second open-weight frontier MoE stack to consolidate a decade of architecture and systems ideas into one release.*

**Related concepts:** [gdla](../architectures/gdla.md) · [differential-attention](../architectures/differential-attention.md) · [polynorm](../architectures/polynorm.md) · [hyper-connections](../architectures/hyper-connections.md) · [mxfp8](../quantization/mxfp8.md) · [multi-teacher-on-policy-distillation](../post-training/multi-teacher-on-policy-distillation.md) · [mla](../architectures/mla.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [mtp](../pre-training/mtp.md) · [fp8-training](../pre-training/fp8-training.md) · [on-policy-distillation](../post-training/on-policy-distillation.md) · [grpo](../post-training/grpo.md)

---

## What this is

Motif 3, released August 2026 by Motif Technologies. A decoder-only Mixture-of-Experts language model: 314B total parameters, 13.2B activated per token. Each MoE layer holds 384 routed experts and selects the top-8 per token — a step further into fine-grained sparsity than DeepSeek-V3's 256 experts. Pre-trained on ~12.5T tokens spanning web, STEM, code, math, multilingual, and domain-specialized corpora, with a post-training pipeline that consolidates six domain-specialist RL teachers plus a software-engineering SFT teacher into one unified model via **Multi-teacher On-Policy Distillation**.

The paper matters because it's the second frontier open-weight tech report (after DeepSeek-V3) to publish a full stack of individually-novel ideas. Four stand out: **GDLA** (Grouped Differential Latent Attention — fuses differential attention with MLA's compressed KV), **PolyNorm activations** made expert-specific inside the MoE, **modified manifold-constrained hyper-connections** for stability at depth, and **selective MXFP8** in compute *and* communication.

---

## Architecture at a glance

```
Decoder-only transformer, MoE
Total params:      314B
Active per token:  13.2B (top-8 of 384 experts)

Attention:  GDLA — Grouped Differential Latent Attention
              differential-attention overlay on MLA-compressed KV
              (see ../architectures/gdla.md and ../architectures/differential-attention.md)

MoE layer:  384 routed experts,  top-k = 8
              expert-specific PolyNorm activations (../architectures/polynorm.md)
              routing: fine-grained per DeepSeek-MoE pattern

Depth mgmt: modified manifold-constrained hyper-connections
              (see ../architectures/hyper-connections.md)

Auxiliary:  Multi-Token Prediction head (../pre-training/mtp.md)
```

All of this is covered in the concept pages — see the **Related concepts** line above.

---

## Training infrastructure

### Numerical format

Selective **MXFP8** in compute and communication — see [mxfp8](../quantization/mxfp8.md). MXFP8 is the MX-format FP8 variant that pairs an 8-bit floating-point mantissa/exponent with a per-block scale, giving finer-grained scaling than DeepSeek-V3's classical per-tile FP8. Communications between GPUs also use MXFP8, cutting all-to-all bandwidth roughly in half at MoE dispatch boundaries. High-precision paths are reserved for embeddings, LM head, MoE gating, normalization, and attention (same "protect the sensitive components" recipe DeepSeek-V3 formalized for standard FP8).

### Context length

Up to **256K tokens** at training time, via memory-efficient fused kernels and **window-aware context parallelism** — the parallelism scheme is tuned to the sliding-window attention pattern GDLA inherits from differential attention.

### Stability techniques

A collection of expert-balancing and numerical-stabilization measures — the paper describes these as engineering rather than research, but the aggregate matters: it lets a 314B / 384-expert MoE train stably in MXFP8 to 12.5T tokens. Modified manifold-constrained **hyper-connections** are the specific structural addition, extending prior work on hyper-connections with a manifold constraint that keeps intermediate representations on a well-conditioned manifold at depth.

---

## Training recipe

### Pre-training

- **~12.5T tokens** across web documents, STEM, code, math, multilingual content, and domain-specialized corpora.
- **MXFP8** selective compute + communication (see above).
- **256K max context** enabled by window-aware context parallelism.
- **MTP** as an auxiliary objective — same role as in DeepSeek-V3, both a quality lift during training and a self-speculation mechanism at inference.
- **Aux-loss-free balancing + sequence-wise guardrail** — inherited from the DeepSeek line (the paper does not claim these as novelties).

### Post-training

This is where Motif 3 breaks from DeepSeek-V3. Instead of running long-CoT RL directly on the frontier model (or, as V3 did, distilling from a single reasoning sibling like R1), Motif 3 trains **six domain-specialist teachers with RL** plus a software-engineering teacher with SFT, then consolidates all seven into the unified base with **Multi-teacher On-Policy Distillation** (see [multi-teacher-on-policy-distillation](../post-training/multi-teacher-on-policy-distillation.md)).

```
General SFT stage
  │
  ├─ 6 specialist teachers (RL)  ─┐
  ├─ 1 SFT-only SWE teacher       ├─── Multi-teacher OPD → unified model
  └─ (base model)                 ─┘
```

The unified model consolidates reasoning, coding, tool use, professional work, long-context understanding, calibrated abstention, and instruction-following in one checkpoint. See the OPD family in [on-policy-distillation](../post-training/on-policy-distillation.md).

---

## Key results

The paper reports competitive performance against leading open-weight models across:

- **Long-horizon agentic tasks** — the specialist-RL-then-distill pipeline is directly aimed at agent capability without destroying general capability.
- **Mathematical reasoning** — one of the six specialist domains.
- **Scientific knowledge** — long-context and structured knowledge tasks.
- **Hallucination-sensitive evaluation** — the recipe includes calibrated abstention as an explicit specialty.

Specific benchmark numbers are in the paper's evaluation section; the report positions Motif 3 as competitive-to-leading among open-weight releases at its parameter budget.

---

## Key takeaways

1. **GDLA proposes MLA and differential attention are complementary, not alternative.** Prior work treated them as separate ideas — MLA for KV compression, differential attention for softmax denoising. Fusing them in one module keeps both wins simultaneously. If GDLA holds up at broader scale, it becomes the default attention primitive for any long-context MoE. See [gdla](../architectures/gdla.md).

2. **Fine-grained MoE keeps scaling.** 384 experts × top-8 pushes past DeepSeek-V3's 256 × top-8; the paper argues expert-specific PolyNorm activation is what makes 384 experts stably specialize. See [polynorm](../architectures/polynorm.md).

3. **Multi-teacher on-policy distillation formalizes an ad-hoc DeepSeek recipe.** V3 distilled reasoning traces from R1; Motif 3 generalizes this to a **portfolio** of RL specialists plus an SFT teacher, all merged into one student. Puts "RL specialists, distill into generalist" on a repeatable footing. See [multi-teacher-on-policy-distillation](../post-training/multi-teacher-on-policy-distillation.md).

4. **MXFP8 generalizes DeepSeek-V3's FP8 recipe.** MX-format's per-block scaling handles wider dynamic ranges more gracefully than classical E4M3 per-tile scaling, especially at MoE routing boundaries where activations can spike. See [mxfp8](../quantization/mxfp8.md).

5. **Manifold-constrained hyper-connections are a stability lever at depth.** As MoE stacks get deeper and sparser, keeping intermediate representations on a well-conditioned manifold matters more; hyper-connections provide the mechanism. See [hyper-connections](../architectures/hyper-connections.md).

6. **Window-aware context parallelism unlocks 256K training.** Sliding-window attention (inherited from differential attention) allows the context-parallelism scheme to shard along a locality-preserving axis, which classical CP schemes don't exploit.

7. **The "second frontier open-weight tech report" bar.** DeepSeek-V3 set the pattern: an open-weight release accompanied by a detailed tech report bundling many individually-significant contributions. Motif 3 is the second at this bar. The pattern should be recognizable now.

---

## What's still opaque

- **Author list and precise institutional attribution** were not surfaced on the HF page at digest time.
- **Detailed hyperparameters** (LR schedule, batch size, teacher/student pairing choices in Multi-teacher OPD) — the paper documents the recipe but the exact scheduling details are in the report's appendix rather than the summary.
- **Which of the four architectural claims (GDLA, PolyNorm, hyper-connections, MXFP8) is doing the most work.** No headline ablation surfaces this; the paper argues for the bundle.
- **Reproducibility trajectory.** No mention of intermediate checkpoint releases (as OLMo 2 provides). Open weights, not open training.

---

*Pairs well with:* the [DeepSeek-V3 case study](deepseek-v3.md) — Motif 3 is the direct architectural descendant, and the delta between the two is the cleanest window we have into how the frontier open-weight MoE stack is evolving.
