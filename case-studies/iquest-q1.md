# Case Study: IQuest-Q1

*A 320B-total / 15B-active sparse MoE agentic foundation model, built from scratch with a three-stage recipe stacking SFT, domain-specialized RL, and multi-teacher on-policy distillation ("MOPD"), followed by model merging across stages. Open-source release targeting agentic CLI and coding workloads.*

**Related concepts:** [_moe](../architectures/_moe.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [mla](../architectures/mla.md) · [mid-training](../pre-training/mid-training.md) · [mtp](../pre-training/mtp.md) · [grpo](../post-training/grpo.md) · [rlvr](../post-training/rlvr.md) · [mopd](../post-training/mopd.md)

---

## What this is

IQuest-Q1 is a sparse-MoE agentic foundation model released October 2026. 320B total parameters, 15B activated per token, pre-trained + mid-trained + post-trained from scratch with a stacked recipe oriented toward CLI-style and coding agent tasks. The paper is a tech report comparable in scope to DeepSeek-V3 or Qwen 2.5: full architecture, training, evaluation, open release.

The interesting part is the composition. The architecture is a modern MoE (fine-grained experts, GQA, hybrid full/sliding-window attention, partial RoPE), but the headline contribution is the end-to-end post-training stack: SFT across two stages, domain-specialized RL per capability axis, Multi-teacher On-Policy Distillation (MOPD) from four frozen expert teachers, and model merging interleaved across stages and expert branches.

---

## Architecture at a glance

```
88 transformer layers
  ├─ layer 1 (dense):        attention + dense FFN (intermediate 12,288)
  ├─ layers 2-86 (MoE):      attention + MoE FFN
  │       - full attention on layers 1-2 and the last two
  │       - middle layers repeat: 3× sliding-window-4096  +  1× full attention
  │       - partial RoPE over 32 of 128 head dims
  └─ output layer:           full attention + MoE FFN

d_model            = 3,072
attention heads    = 48 Q, 8 KV  (GQA 6:1)
d_head             = 128
rotary dims        = 32  (partial RoPE)
sliding window     = 4,096

MoE per layer      = 256 routed experts, top-8 activated per token
                     expert intermediate dim = 1,536  (fine-grained)
                     layer 1 uses a dense FFN instead

vocab              = 160,000  (untied input/output embeddings)
params             = 320B total, 15B activated per token
MTP drafter        = recursive, 7 drafting steps (the Table 1 "×8" is a documented inconsistency)
```

---

## Training recipe

### Pre-training

- **Two-stage curriculum.** Stage 1 is a broad corpus; stage 2 increases the share of code data.
- **Normalization.** Stable SplitNorm, adopted from prior work and extended with (a) RMSNorm on the attention output and (b) SandwichNorm on the first layer. The resulting block is `z = RMSNorm(x); h = z + RMSNorm(Attention(z)); y = h + FFN(RMSNorm(h))` for layers after the first.
- **Optimizer.** Muon.
- **Parameterization.** muP for hyperparameter transfer across scales.
- **Load balancing.** Aux-loss-free bias control on the routed experts (standard modern MoE default).

### Mid-training (progressive context extension)

Context length is extended through three successive stages: 32K → 128K → 512K, each a dedicated training phase. Data mix broadens across domains and formats and shifts to higher-quality agentic / tool-use samples.

### Post-training (the stacked part)

A four-method sequence, applied in order:

1. **SFT Stage 1.** Instruction + reasoning + agentic traces. Produces the base post-trained checkpoint used as the starting point for RL branches and MOPD.
2. **Domain-specialized RL.** Separate RL runs per capability axis (agentic UX, multi-harness, long-horizon, general agentic) produce **four specialist teacher checkpoints**. RL is GRPO-family with rule-based rewards on verifiable tasks and model-based rewards elsewhere ([RLVR](../post-training/rlvr.md)).
3. **SFT Stage 2.** Starts from a merged checkpoint; refines on curated SFT data.
4. **MOPD (Multi-teacher On-Policy Distillation).** The student (SFT Stage 2 checkpoint) generates rollouts; each task is routed to the matching frozen specialist teacher, which scores the student's tokens. Advantage is the clipped log-ratio `log(q_teacher / p_student)`; loss is a length-normalized reverse-KL surrogate. Tool observations and instructions are masked from the loss, and repetitive rollouts are filtered. See [mopd](../post-training/mopd.md).
5. **Model merging.** Interleaved across stages and across expert branches. Each merge computes parameter deltas against a reference checkpoint, combines them with prescribed weights and global ℓ2-norm calibration, and adds the result back to the reference. The paper does not publish the exact weighting formulas.

---

## Key results

Reported scores (authors' own evaluation; some baselines re-run by the authors):

| Benchmark | IQuest-Q1 |
|---|---|
| SWE-Bench Pro | 57.5 |
| SWE-Bench Multilingual | 80.3 |
| DeepSWE v1.1 | 64.6 |
| LiveCodeBench v6 | 89.7 |
| Terminal-Bench 2.1 | 83.2 |
| CyberGym | 84.5 |
| GPQA Diamond | 93.3 |
| IMOAnswerBench-Verified | 93.0 |
| Humanity's Last Exam (no tools / with tools) | 39.2 / 55.7 |
| BrowseComp | 85.5 |
| Toolathlon-Verified | 75.0 |

Headline positioning is agentic / coding workloads; the model skips HumanEval, MMLU, AIME in favor of harder agent-flavored benchmarks.

---

## Key takeaways

1. **The frontier agentic recipe is stacked, not single-trick.** SFT → RL → more SFT → MOPD → model merging layered on top of fine-grained MoE pretraining. Any single stage would be a paper; the composition is the point.
2. **MOPD matters.** On-policy distillation from several RL-trained specialists into one general student lets the final model inherit multi-capability expertise without conflicting RL objectives — all the capability composition happens in distillation space, not in RL loss space. See [mopd](../post-training/mopd.md).
3. **Fine-grained + top-8 routing is the modern MoE default.** 256 routed experts with small (1,536) intermediate dim, top-8 activated, continues the DeepSeek-MoE lineage.
4. **Progressive context extension is a staged training phase, not a one-shot RoPE adjustment.** 32K → 128K → 512K each gets its own training phase with data-mix changes, not just a scaling tweak.
5. **Partial-RoPE + hybrid attention** (3× SWA-4096 + 1× full, with RoPE on only 32 of 128 head dims) is the production choice for long context on an agentic model. Keeps costs sane at 512K.
6. **Model merging is now an interleaved production step**, not a one-shot post-hoc trick. Delta-against-reference with ℓ2-norm calibration is applied between SFT and RL stages and across the expert branches.

---

## What's still opaque

- **Model-merging formulas** are not published; delta-weights and ℓ2-calibration constants are prescribed but not stated. Full reproduction is blocked here.
- **Headline benchmarks on standard LLM evals** (MMLU, HumanEval, AIME) are not reported. Agentic-focused evaluation makes direct comparison with DeepSeek-V3 / Qwen 2.5 harder.
- **MTP head "×8 vs seven" inconsistency** between the text and Table 1 is left unresolved in the paper.
- **Training cost** is not disclosed.
- **Data mixture** at each pre-training and mid-training stage is described qualitatively rather than quantitatively.
- **Safety / refusal training** is not characterized.

---

*Pairs well with:* the [DeepSeek-V3 case study](deepseek-v3.md) for direct architectural comparison (both are fine-grained MoE at frontier scale, with different attention and context-extension recipes).
