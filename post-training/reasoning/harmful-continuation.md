# Harmful Continuation (in Long-CoT SFT)

*Depth — answer-correct long-CoT traces whose post-conclusion tail degrades supervised fine-tuning.*

**TL;DR:** Distilling a long-CoT reasoner via SFT on its answer-correct traces is the standard recipe (R1, Kimi, the open distill ecosystem). But "answer-correct" is not sufficient. Many correct traces include a *post-conclusion continuation* — reasoning that runs past the point where the answer is supported. SFT'ing on those tails hurts the student, because they pair persistent local uncertainty with no terminal-directional progress in hidden-state geometry. The Harmful Continuation Cut (HCC) detects the boundary and truncates.

**Prereqs:** [long-cot-rl.md](./long-cot-rl.md), [../rejection-sampling.md](../rejection-sampling.md)
**Related:** [length-penalty.md](./length-penalty.md), [long2short.md](./long2short.md), [../../case-studies/deepseek-r1.md](../../case-studies/deepseek-r1.md), [../../case-studies/kimi-k1-5.md](../../case-studies/kimi-k1-5.md)

---

## What it is

A long-CoT trace produced by an RL-trained reasoner is rarely *terminated* at the moment the answer becomes supported. The model often continues for hundreds of tokens — restating, double-checking, second-guessing — before emitting the final answer. Two regimes within answer-correct traces:

| Region | Local uncertainty | Hidden-state direction | SFT effect |
| --- | --- | --- | --- |
| Pre-conclusion | High → low | Steady progress toward answer | Useful — teaches the reasoning trajectory |
| **Post-conclusion continuation** | Persistently low-but-nonzero | Stalled / wandering | **Harmful — teaches the model to never commit** |

The He et al. (UESTC / SUTD / SMU 2026) paper uses a delete-only editor on a held-out probe set to identify the boundary, then compares SFT on the original vs trimmed traces — trimming wins, confirming the harm hypothesis.

## How it works

The diagnostic: each long-CoT trace is paired with a *suffix-removal* version where everything past the post-conclusion boundary is deleted (answer preserved). SFT on the trimmed dataset consistently outperforms SFT on the original.

The mechanism — *uncertainty–geometry mismatch*:

- **Persistent local uncertainty.** Per-token entropy stays at a non-zero plateau in the continuation tail — the model hasn't "committed".
- **Weakened terminal-directional progress.** Project hidden states onto the direction toward the final-answer hidden state. Pre-conclusion: monotone progress. Post-conclusion: stalled or noisy.

When both conditions hold simultaneously, the trace is in the harmful tail.

**HCC** (Harmful Continuation Cut) is a lightweight boundary proxy that approximates the editor's decision without running a heavy verifier:

```
for each long-CoT trace:
    for each token position t:
        u_t = entropy(model_logits[t])
        p_t = (hidden[t] - hidden[0]) · direction_to_answer
        if u_t > τ_u  and  d/dt(p_t) < τ_p:
            mark t as post-conclusion candidate
    boundary = first sustained candidate region
    truncate trace at boundary
```

## Why it matters

- **R1-style distillation just got a quality knob.** The 800k-sample R1 resample, Kimi's long-CoT SFT, the open community's R1-distill datasets — all rely on the "answer-correct" filter. HCC adds a tail-trim step that consistently improves the downstream student.
- **Length isn't directly the bug.** Length is the symptom; the model continuing past commitment is the disease. That's why a length penalty alone underperforms HCC — it cuts both useful exploration and harmful wandering equally.
- **No new model needed.** Boundary detection runs on the *student's* own hidden states + entropies — no separate verifier or judge.

## Gotchas & tricks

- **Threshold tuning per source.** Different RL-trained teachers have different continuation profiles; the entropy threshold $\tau_u$ and direction-progress threshold $\tau_p$ benefit from per-source calibration on a held-out set.
- **Don't truncate the answer.** HCC must respect answer-token boundaries; truncating into the answer span destroys the example.
- **Compose with rejection sampling.** HCC is orthogonal to verifier-correct filtering. R1 Stage 3 + HCC ≈ correct-and-cleanly-terminated.
- **Less useful for short-CoT.** The harm comes from the long tail; short reasoning traces have less continuation to remove and gains shrink.

## Sources

- *Diagnosing Harmful Continuation in Answer-Correct Long-CoT Training Traces* — He et al., UESTC / SUTD / SMU, 2026 — [arXiv:2605.29288](https://arxiv.org/abs/2605.29288) — primary source for the harmful-continuation phenomenon, the editor study, and HCC.
- *DeepSeek-R1* — DeepSeek, 2025 — Stage 3 answer-correct SFT pipeline that HCC sharpens.
