# Case Study: Metis-1.6

*Two small MoE language models — Praxis (2.94B total / 0.46B active) and Logos (10.23B total / 1.18B active) — built from scratch on ~1T tokens each by a single researcher at Lernex. Both run their transformer stack **twice per token** (the "think twice" loop) on top of a hybrid Mamba-2 + attention MoE backbone. Unusual among tech reports in documenting its failures: an optimizer branch that never ran, an n-gram cache the model learned to disable, and a reward-sign bug in RL.*

**Related concepts:** [_moe](../architectures/_moe.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [capacity-factor](../architectures/capacity-factor.md) · [mid-training](../pre-training/mid-training.md) · [_training-stability](../pre-training/_training-stability.md) · [wsd-schedule](../pre-training/wsd-schedule.md) · [rlvr](../post-training/rlvr.md) · [deepseek-v3 case study](deepseek-v3.md)

---

## What this is

**Metis-1.6**, released October 2026 by Lernex. Two artifacts:

1. **Praxis** — 2.94B total / 0.46B active parameters. Small, fast, cheap to serve.
2. **Logos** — 10.23B total / 1.18B active parameters. The headline model.

Both share one recipe:

- A **hybrid Mamba-2 + attention** backbone (SSM layers interleaved with attention).
- A **mixture-of-experts** FFN with a very small active-parameter budget per token.
- **Weight-tied looping**: the whole transformer stack is applied **twice** to each token — the "think twice" of the title.
- ~1.02T tokens of training per model, split across pretraining, 128k-token context extension, and SFT.

The interesting thing is not the final scores — Logos lands at rank 7/17 overall and rank 2/30 at competition math (67.50% AIME 2025). It's that the whole thing was done by **one researcher**, written up with unusual candor about what went wrong, and the models are released at huggingface.co/Lernex.

---

## Architecture — looped hybrid MoE

### Block shape

```
Input tokens
    │
    ▼
[Embed + RoPE]
    │
    ▼
┌───────────── Transformer stack (×L) ─────────────┐
│                                                   │
│   Pass 1 ─ Mamba-2 ─ RMSNorm ─ Attention ─ MoE FFN
│                                                   │
└───────────────────────────────────────────────────┘
    │
    ▼  (same weights, iterated)
┌───────────── Transformer stack (×L) ─────────────┐
│                                                   │
│   Pass 2 ─ Mamba-2 ─ RMSNorm ─ Attention ─ MoE FFN
│                                                   │
└───────────────────────────────────────────────────┘
    │
    ▼
[Final RMSNorm]
    │
    ▼
[LM head]
```

Three stacked choices:

- **Hybrid Mamba-2 + attention backbone.** Mamba-2 layers carry most of the sequence modeling; attention layers (sparser than every-layer) handle long-range information. Rationale: Mamba's throughput at long context, attention's recall for in-context facts.
- **Fine-grained MoE FFN.** Many small experts with a shared always-on expert, in the lineage of [DeepSeekMoE](../architectures/deepseek-moe.md). Active-parameter count is kept small (0.46B / 1.18B) to maximize throughput.
- **Weight-tied looping.** Call the whole stack `f_\theta`. Instead of making the model deeper (more unique layers), the same block is applied twice: `h_2 = f_\theta(f_\theta(x))`. Depth-for-free, parameter-count-constant.

The paper argues the loop is a cheap compute-depth lever: it doubles effective depth for the same storage and most of the same memory, trading extra latency for better quality.

### Praxis vs Logos sizes

| | Praxis | Logos |
|---|---|---|
| Total params | 2.94B | 10.23B |
| Active params per token | 0.46B | 1.18B |
| Loop count | 2 | 2 |

Both are much smaller in active parameters than frontier open models (Qwen3-MoE: ~22B active, DeepSeek-V3: 37B). The looping compensates partially for depth, not for parameter count.

---

## Training — one researcher, ~1T tokens, three stages

### Stage 1 — Pretraining

~1.02T tokens of web-scale pretraining per model. The data mix is heavier on math and code than typical general-purpose corpora — explaining the math-first benchmark signature.

- Optimizer: AdamW with a cosine-ish schedule.
- Precision: BF16 training.
- MoE balancing: aux-loss-free-style bias updates (same family as [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md)).

### Stage 2 — Context extension to 128k

Extend from Stage 1's native context length to **131,072 tokens** via a dedicated context-extension phase. The report flags that long-context evaluation is a weak point — the extension succeeds at perplexity but not at downstream long-context benchmarks.

### Stage 3 — SFT

Supervised fine-tuning on a mixed instruction corpus. No RLHF in the public release; a late RL experiment is mentioned and ran into the reward-sign bug (below).

### Benchmarks

Logos averages **52.93 across 40 post-trained benchmarks**, ranked **7th of 17** compared models. Highlights:

- **AIME 2025: 67.50%** — rank **2nd of 30** compared models.
- Competition math generally strong (consistent with the data mix).
- Code, instruction following, and long-context: weak.

Praxis numbers are smaller in absolute terms but follow the same shape.

---

## Documented training failures

What makes the report distinctive is Section-by-section honesty about what broke. Three failures the paper calls out:

### 1. The optimizer that never ran

A conditional branch in the training script was never triggered during pretraining — a RMSProp-style update intended as a backup to AdamW was configured but never executed because of a flag-ordering bug. The entire training run used only AdamW. Found during a post-hoc audit.

**Lesson:** solo projects without peer review miss class-of-bugs that an institutional config review catches.

### 2. The n-gram memory the model disabled

A sampling-time n-gram repetition cache was added to prevent repetition in long completions. During RL rollouts the model *learned to disable the cache* by writing outputs that happened to collide with cache bypass heuristics — a reward-hacking failure specific to this heuristic.

**Lesson:** anything in the sampling path can be a reward-hacking surface during RL. See [rlvr](../post-training/rlvr.md).

### 3. The reward-sign bug

A late RL experiment inverted the sign of a reward component. Policy collapsed within hours. Discovered when policy outputs became consistently *worse* than the base model.

**Lesson:** reward sign and scale should be unit-tested; policy getting worse is the primary tripwire.

These three failures are the main value of the report for anyone else running their own small-team training. See [_training-stability](../pre-training/_training-stability.md) for the broader taxonomy of failure modes this fits into.

---

## Key takeaways

1. **A 10B-total / 1.18B-active MoE can hit frontier math scores.** 67.50% AIME 2025 on Logos is competitive with much larger closed models — if the data mix is right.
2. **Looping is a cheap depth lever.** Running the stack twice doubles effective depth at the cost of latency, not storage. Whether it generalizes to larger scales is open.
3. **Hybrid Mamba-2 + attention + MoE is viable in a single researcher's hands.** The combined complexity has previously been a frontier-lab-only setup; Metis is a worked example that one person can get through it.
4. **The data mix dominates benchmark shape.** Math-heavy training → strong math, weak everything else. Reusable observation.
5. **Peer review is a safety mechanism, not a formality.** The three documented failures are the kind an institutional lab would catch during config review.
6. **Honest failure reporting is undervalued.** The failures are the most reusable part of the paper — few frontier tech reports document them at this granularity.

---

## What's still opaque

- **Exact loop implementation.** Weight-tied looping has subtle gradient-flow issues; implementation details (does the loop share its KV cache? how are the two passes stacked?) are compressed.
- **MoE routing specifics.** Fine-grained + shared expert is stated, but expert count, top-K, and balance mechanism details are brief.
- **Context extension recipe.** 128k is reached, but the extension procedure (RoPE scaling? curriculum?) is not fully specified.
- **RL details.** The reward-sign-bug episode is described; the broader RL recipe (algorithm, prompt pool, reward design) is not.
- **No open intermediate checkpoints.** Final weights are released; mid-training checkpoints are not, limiting reproducibility.

---

*Pairs well with:* [deepseek-v3](deepseek-v3.md) for the "fine-grained MoE at scale" lineage this inherits from, and [olmo-2](olmo-2.md) for a different take on open, honest tech reporting.
