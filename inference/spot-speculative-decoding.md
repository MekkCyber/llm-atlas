# SPOT — Speculative Decoding for Diffusion LMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The first working speculative decoding for **diffusion language models (DLMs)**. Because DLMs must choose both *what* token to commit and *where* (which position) at each step, standard AR speculative decoding doesn't apply. SPOT trains a tiny (0.62M-param) drafter — called the **Speculating Positions Order Trajectory** drafter — that predicts the commit order from the target model's own output distributions. The drafter is cheap enough to run inside one target diffusion pass, enabling up to **6.8× fewer target passes** and **up to 3.4× higher throughput** on SDAR-8B. A variant, **SPOT-Sets**, extends to dynamic-threshold decoding.

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md)
**Related:** [_speculative-decoding.md](_speculative-decoding.md) · [../multimodal/unified-diffusion-llm.md](../multimodal/unified-diffusion-llm.md)

---

## What it is

AR speculative decoding predicts *the next tokens*. DLMs process all positions in parallel at each diffusion step and commit the most confident subset — committing both *what* and *where*. A drafter that only proposes next tokens doesn't tell you where to commit them, and committing in the wrong order can produce different final text.

SPOT handles both. It observes that the target DLM's output distributions from the **previous diffusion pass** already contain almost all the information needed to pick the next commits: the token prediction is correct at over **90% of positions**, and the confidence pattern across positions reveals the right commit order. A tiny drafter learns to extract that order prediction cheaply.

## How it works

### Standard DLM step (for context)

At diffusion step $t$:

1. Forward pass over all positions → distribution $p_t(x_i)$ per position $i$.
2. Pick positions with high confidence → commit them this step.
3. Mask the rest, iterate.

Cost per step is one full forward pass.

### SPOT step

```
previous target pass → cached output distributions p_{t-1}(x_i)
                                   │
                 ┌─────────────────┴──────────────────┐
                 ▼                                    ▼
         SPOT drafter (0.62M params)         target DLM forward pass
         predicts: which k positions          verifies drafter's
         to commit and in what order          proposals
                 │                                    │
                 └──────────────┬─────────────────────┘
                                ▼
             accept longest matching commit prefix,
             fall back to standard commit at first mismatch
```

The drafter is a small MLP/transformer trained on target-model behavior. Its input is the previous pass's output distribution; its output is a predicted commit trajectory (ordered list of positions + their tokens). The target pass verifies: each predicted commit is accepted iff the target's own commit choice at that step would be the same.

### SPOT-Sets

For DLMs that use **dynamic thresholds** (commit anything with confidence above a per-step threshold rather than a fixed count), the drafter emits a set-valued prediction instead of an ordered list. Verification uses the set membership against the target's threshold choices.

## Why it matters

- **Opens speculative decoding to DLMs.** Prior to SPOT, DLMs had no working speculative-decoding method — they were a serving-cost liability compared to AR models.
- **Big real speedup.** Up to **6.8× fewer target passes** and **up to 3.4× higher throughput** on SDAR-8B. The drafter is small enough that its overhead is negligible.
- **Reuses existing passes.** The drafter's input is the *previous pass's* output distribution — information that was already computed and would otherwise be discarded.
- **Validates [unified diffusion LLMs](../multimodal/unified-diffusion-llm.md).** Combined with SPOT, discrete diffusion backbones become competitive on serving cost with AR models.

## Gotchas & tricks

- **Distribution preservation.** Acceptance is defined against the target's own commit mechanics, so the final output distribution is preserved up to the DLM's own stochasticity — not just approximated.
- **Drafter is per-target.** Trained on one target DLM. Switching the target requires retraining the drafter (though it's small).
- **The 90% figure is key.** The paper reports that the target's prediction from the previous pass is already the right token at >90% of positions. SPOT's win depends on that overlap holding on the deployed workload.
- **SPOT-Sets vs SPOT.** Use SPOT-Sets when the target uses dynamic-threshold commits (common in recent DLMs); SPOT when it uses fixed-count commits.
- **Compute asymmetry.** DLM inference is often bandwidth-bound like AR; SPOT's drafter is cheap enough that the main cost stays the target pass. Rejected proposals waste only the drafter cost, which is negligible.
- **Not applicable to continuous flow models.** The commit-order structure is specific to discrete diffusion. Continuous flow has no commit step to speculate.

## Sources

- Paper: *On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction* — Shlomi, Rahimi, Mendelson, Baskin, 2026 — [arXiv:2610.22543](https://arxiv.org/abs/2610.22543)
