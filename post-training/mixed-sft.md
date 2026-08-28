# Mixed SFT
*Depth — jointly fine-tuning on no-CoT and long-CoT data in one supervised stage, as a strong alternative to next-chunk reasoning RL.*

**TL;DR:** Next-chunk reasoning RL was pitched as the way to extract reasoning signal from *no-CoT corpora* (worked solutions, textbook derivations that lack explicit chain-of-thought). Tang et al. (2026) show that a much simpler baseline — one SFT stage that jointly fine-tunes on both no-CoT and long-CoT data — reaches a higher **post-RLVR** performance ceiling at over **60× less** training compute.

**Prereqs:** [rlvr.md](rlvr.md), [_post-training.md](_post-training.md)
**Related:** [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

"No-CoT data" is corpora rich in worked reasoning content — textbook derivations, solutions, proofs — but without explicit chain-of-thought annotations connecting question to answer. Getting reasoning signal out of it has been treated as a training-design problem: prior work introduced *next-chunk reasoning RL*, which trains a model to generate implicit reasoning traces and rewards them by their ability to predict the next chunk of text.

Mixed SFT is the missing simpler baseline: **a single supervised fine-tuning stage that jointly trains on no-CoT and long-CoT data**, mixed within the same batch, using standard next-token cross-entropy. No RL, no chunk-level reward, no additional stages.

## How it works

Two data streams merged into one SFT run:

1. **Long-CoT examples** with full reasoning traces — teach the model the *shape* of reasoning.
2. **No-CoT examples** (worked solutions, derivations, textbook problems with only final answers or terse solutions) — teach the model the *content* of the domain.

The two streams are mixed at the batch level; loss is standard token-level cross-entropy on both. That's it — no auxiliary objective, no chunk-level reward, no separate stages.

The load-bearing methodological point is that the fair comparison is not "next-chunk RL vs plain SFT on long-CoT data alone" but **through the full post-training pipeline**: SFT stage → RLVR stage → measure the final result. Mixed SFT beats next-chunk reasoning RL on the post-RLVR ceiling for both in-domain math and out-of-domain reasoning tasks.

## Why it matters

Two takeaways:

1. **Simpler wins.** A trending RL-flavored line of work is beaten by the naive mixed-corpus SFT baseline at >60× less compute. This is worth remembering when new RL-shaped training tricks arrive without a plain-SFT ablation.
2. **Evaluate post-RLVR, not pre-RLVR.** Higher pre-RLVR accuracy does *not* necessarily translate to higher post-RLVR accuracy. Comparing methods only at the SFT boundary can pick the wrong winner. Any SFT-vs-RL claim about no-CoT data must be evaluated after the full post-training pipeline has run.

## Gotchas & tricks

- **Mixture ratio matters, but modestly.** The paper's finding is robust across reasonable no-CoT / long-CoT mixture ratios; there's no cliff, but very extreme ratios in either direction hurt.
- **Only meaningful with RLVR downstream.** The comparison assumes an RLVR stage follows SFT. If you plan to ship the SFT checkpoint directly, the picture may differ — pre-RLVR and post-RLVR rankings are not the same.
- **No-CoT ≠ noisy CoT.** Mixed SFT is about *absence* of chain-of-thought in the source data, not corrupted CoT. Textbook derivations and worked solutions qualify; garbled reasoning traces don't.

## Sources

- Paper: *Is Next-Chunk Reasoning RL Really Better than SFT? Revisiting Training Strategies under no-CoT Data* — Tang et al., 2026 — [arXiv:2608.23256](https://arxiv.org/abs/2608.23256)
