# long2short — Distilling Long-CoT into Short-CoT
*Depth — four methods for transferring a long-CoT reasoning model's capability into a token-efficient short-CoT model.*

**TL;DR:** After training a long-CoT model via [long-cot-rl](long-cot-rl.md) or similar, you have strong reasoning but expensive inference (thousands of tokens per response). **long2short** is the class of methods that compress the capability into a short-CoT model. Kimi k1.5 compares four: **(a) model merging** (weight-average long + short model), **(b) shortest rejection sampling SFT** (resample `n=8`, SFT on the shortest correct), **(c) DPO** on `(shortest correct, long-or-incorrect)` pairs, **(d) long2short RL** (a second RL phase with a length-capped max rollout plus the length penalty). Long2short RL wins on token efficiency: **Pass@1 = 60.8 on AIME 2024 with ~3,272 tokens avg** — a frontier short-CoT result.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [length-penalty](length-penalty.md), [rejection-sampling](../rejection-sampling.md), [dpo](../dpo.md), [model-souping](../../pre-training/model-souping.md)
**Related:** [online-policy-mirror-descent](online-policy-mirror-descent.md) · [kimi-k1-5 case study](../../case-studies/kimi-k1-5.md)

---

## What it is

A set of post-processing methods applied to a trained long-CoT reasoning model. Input: a long-CoT model (strong but verbose). Output: a short-CoT model (ideally just as strong, much faster at inference). Introduced by Kimi k1.5 (Sec. 2.4, Sec. 3.4).

Why it matters: long-CoT RL produces models that emit 4k–20k-token reasoning traces per response. For production inference this is expensive (4×–20× the tokens of a normal assistant response). Traditional inference-time-compute arguments say "long is worth it for the hard problems" — but for the majority of queries where a shorter response suffices, you want a model that defaults to short.

long2short is the bridge. Four methods, increasing cost and capability transfer:

| Method | Cost | Training type | Key idea |
|---|---|---|---|
| **Model merging** | ~free | weight average | Linearly interpolate long-CoT model weights with a short-CoT model |
| **Shortest rejection sampling SFT** | 1 SFT pass | supervised | Sample `n=8`, pick shortest correct, SFT on it |
| **DPO on length pairs** | 1 DPO pass | offline pref | Positive = shortest correct; negatives = incorrect + too-long correct |
| **long2short RL** | 1 RL phase | online RL | Second RL phase with length-capped rollouts + length penalty |

Kimi reports long2short RL is the best token-efficient method.

---

## How it works

### Method (a): Model merging

```
θ_short = (1 − α) · θ_long  +  α · θ_short_base
```

for some mixing weight `α`. Works because nearby minima (shared initialization or shared early training) are linearly connected — averaging their weights lands on a reasonable point between their behaviors. Details and failure modes: [model-souping](../../pre-training/model-souping.md).

- **Cost**: single state-dict average. Zero training.
- **Assumption**: the long-CoT and short-CoT models share enough parameter-space proximity. In practice, both should be fine-tuned from the same base; directly merging a long-CoT model with an unrelated short-CoT model is less predictable.
- **Quality**: weakest of the four methods per k1.5's Fig. 7. The merged model is "in between" — neither fully long nor fully short.

### Method (b): Shortest Rejection Sampling (SRS) SFT

```
for each prompt x:
    {o_1, ..., o_8} ~ π_long(x)                # sample n=8 responses
    correct = [o_i for o_i in responses if r(x, o_i) = 1]
    if correct is non-empty:
        o_target = arg_min(len(o) for o in correct)
        sft_data += [(x, o_target)]

π_short = SFT(π_base, sft_data)
```

Pick the shortest correct response per prompt as the SFT target. Same pattern as [rejection-sampling](../rejection-sampling.md) but with `len(o)` as the tie-breaker among correct responses.

- **Cost**: `n=8` rollouts per prompt + one full SFT pass.
- **Quality**: middle. Better than merging (targeted distillation); worse than long2short RL (SFT is one-pass, no optimization pressure on length continuously).
- **Knobs**: `n` (rollout count), the correctness filter, any diversity filters on top.

### Method (c): DPO on length pairs

Build preference pairs from the long-CoT model's rollouts:

```
for each prompt x:
    samples = π_long(x)   sampled multiple times
    correct = [s for s in samples if r(x, s) = 1]
    incorrect = [s for s in samples if r(x, s) = 0]
    if correct is non-empty:
        y_w = arg_min(len(s) for s in correct)          # shortest correct = preferred
        negatives = incorrect + [s for s in correct if len(s) > 1.5 · len(y_w)]
        for y_l in negatives:
            dpo_data += [(x, y_w, y_l)]

π_short = DPO(π_base, dpo_data)
```

Key structural choices:
- **Positive** = shortest correct.
- **Negatives** = all incorrect + all correct responses **>1.5× the length** of the positive. (Correct-but-too-long also gets dispreferred — the policy learns both "be correct" and "be concise".)
- Train via standard [dpo](../dpo.md) on these pairs.

- **Cost**: rollouts + one DPO pass.
- **Quality**: middle-to-strong per k1.5's Fig. 7; worse than long2short RL but better than SRS in many settings.

### Method (d): long2short RL

Second RL phase on top of the best checkpoint from the main RL run:

```
π_0 = best_long_cot_checkpoint                  # checkpoint near length/perf peak
Hyperparameters changed from main RL:
  - length_penalty: enabled at constant weight (no warm-up disable)
  - max_rollout_length: significantly reduced
Run RL for one more phase using online-policy-mirror-descent or GRPO.
π_short = final checkpoint
```

Key points:
- Starts from a **near-optimal long-CoT checkpoint** — not from scratch.
- **Length penalty on from step 0** (see [length-penalty](length-penalty.md)).
- **Tighter max-length cap** — even correct responses longer than the cap are truncated and may still pay the length penalty.
- Algorithm: [online-policy-mirror-descent](online-policy-mirror-descent.md) (for k1.5). Substitute [GRPO](../grpo.md) if you're reproducing.

- **Cost**: one more RL phase (10–30% of the main run).
- **Quality**: best of the four. k1.5's Fig. 7 shows long2short RL strictly dominating the others on the token-vs-accuracy frontier.

### Reported results

From Kimi k1.5 Sec. 3.4:

- **k1.5-short w/ rl** (long2short RL method): **Pass@1 = 60.8 on AIME 2024** (avg over 8 runs) using **~3,272 tokens** average per response.
- **k1.5-shortest**: **Pass@1 = 88.2 on MATH500** at comparable token budget to other short-CoT models.

For reference, long-CoT k1.5 hits 77.5 on AIME 2024 at much longer token budgets. long2short RL preserves most of the capability at a fraction of the tokens — the cleanest quantitative result in the paper.

Note: per-method head-to-head tables are not given in the paper — only Fig. 7 (token-vs-accuracy scatter). The ordering "long2short RL > DPO > SRS > merge" comes from that plot.

### Model naming convention

Kimi k1.5's paper names the variants (Sec. 3.4):
- `k1.5-long` — the main long-CoT model.
- `k1.5-short w/ rl` — long2short RL output.
- `k1.5-short w/ dpo` — DPO output.
- `k1.5-short w/ merge` — model-merge output.
- `k1.5-short w/ merge + rs` — merge followed by rejection-sampling SFT.
- `k1.5-shortest` — the most aggressive short-CoT variant.

---

## Why it matters

- **Distillation path that doesn't need a teacher model.** Classical distillation requires a larger teacher. long2short takes a model and compresses *itself* — the long-CoT model is its own teacher for the short-CoT model.
- **Production-friendly output.** Long-CoT models emit expensive traces; most real users don't want to wait 30s for an answer. long2short produces deployable short-CoT models at near-frontier quality.
- **Reusable primitives.** The four methods map to four different tools (merging, SFT, DPO, RL) — useful to understand each separately because they all apply beyond length compression (e.g., distilling any behavior, compressing any capability).
- **Established pattern**. R1 skipped this; the R1 paper focuses on long-CoT and defers short-CoT to the general distillation story (SFT on R1 traces). Kimi's long2short is the first formal study of the problem with comparative methods. Expect future reasoners to include similar steps.

---

## Gotchas & tricks

- **Start from a near-optimal long-CoT checkpoint.** If you start long2short RL from an undertrained long-CoT model, you lock in both short length *and* mediocre reasoning. The long-CoT phase is prerequisite.
- **Length-penalty ramp-up still matters.** Even in long2short RL, turning the penalty on too aggressively early can collapse reasoning quality. The paper applies it from step 0 in long2short but with careful max-length / coefficient tuning.
- **Correctness filter is load-bearing.** SRS and DPO both need a reliable "correct vs incorrect" signal to pick the shortest-*correct*. For non-verifiable tasks (general chat), you need a judge model — which introduces its own hacking surface.
- **Model merging requires shared ancestry.** Merging `θ_long` with a differently-lineaged short model rarely works. Safer: both models fine-tuned from the same base, same recipe, different training budget.
- **DPO's negatives need care.** The 1.5× length threshold is a paper choice; other thresholds (1.2×, 2×) change how aggressive the length shortening is. Too aggressive → penalizes correct reasoning that legitimately needs length. Too loose → no learning signal.
- **`n = 8` is a small sample.** For hard prompts where the long-CoT model has low success rate, 8 samples might yield zero correct responses and the prompt gets dropped. Mitigation: higher `n` for hard prompts, or curriculum.
- **Token count is tokenizer-specific.** "3,272 tokens average" under Kimi's tokenizer isn't the same real-world length under a different tokenizer. Compare within-tokenizer.
- **Pass@1 alone is insufficient.** long2short methods can shift the accuracy/length trade-off smoothly. Report pass@1 **jointly with average tokens** (like Fig. 7) to compare fairly.
- **long2short RL is not a free extra boost on top of the main RL.** It's a specialization phase that trades length for (usually) a small accuracy drop. Don't expect "long2short gives you frontier scores at short-CoT cost with no trade-off".

---

## Sources

- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI (Kimi Team), 2025, arXiv 2501.12599 — introduces long2short (Sec. 2.4) and reports the four-method comparison (Sec. 3.4, Fig. 7).
- Related: [rejection-sampling](../rejection-sampling.md) — SRS is a rejection-sampling-SFT variant with length-as-tiebreaker.
- Related: [dpo](../dpo.md) — the DPO variant uses the standard DPO loss with length-structured preference pairs.
- Related: [model-souping](../../pre-training/model-souping.md) — the model-merging variant reuses classical weight-averaging.
- Related: *DeepSeek-R1* — 2025, arXiv 2501.12948 — contrasting case: R1's short-CoT variants come from distilling R1 traces into smaller models, not from a long2short post-processing phase on R1 itself.
