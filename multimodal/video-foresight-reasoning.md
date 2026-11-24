# Video Foresight Reasoning (Future-OPSD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a video-LLM to **predict what happens next** and plan from the prediction, using only **raw past-future video pairs** — no labels, no judge LLM. The loop is **on-policy self-distillation** gated by a learned "foresight score" that weights frames by decision-relevance. +14.2 open-ended prediction and +15.9 closed-ended planning over supervised baselines.

**Prereqs:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md), [rlvr](../post-training/rlvr.md)
**Related:** [_rl](../post-training/_rl.md)

---

## What it is

Reasoning RL in text uses verifiable rewards (RLVR): a problem has a ground-truth answer, so rollouts can be scored. In video, ground-truth future-frame content is hard to label and existing judge-LLM pipelines are expensive and noisy.

Future-OPSD observes that **naturally occurring video already carries the signal**: the actual next segment of the same video *is* the ground-truth future. So the training loop uses past-segments as input, policy samples predictions for the future, and compares against the real future segment. The comparison is weighted by a **foresight score** that upweights the frames most useful for planning decisions.

---

## How it works

**Past-future pair sampling.** For each video, split into a "past" window (input) and a "future" window (target). The model predicts future content (in a captioning / planning format); scoring compares the prediction against the real future.

**Foresight score.** A learned head, trained jointly, weights each frame of the future by how much it would change downstream decisions. Early boring frames get low weight; frames with state changes, interactions, or visual surprises get high weight. The score sharpens the training signal — the policy is rewarded for getting *decision-relevant* future content right, not every frame equally.

**On-policy self-distillation.** The policy rolls out its own future predictions, scores them against the (foresight-weighted) ground-truth future, and distills the high-scoring rollouts back into itself. This is RL-style on-policy sampling without an external reward model — the "reward" is the match against the real video continuation.

**No external supervision.** No action labels, no annotated decision traces, no judge LLM. Everything comes from naturally occurring video temporal structure.

---

## Why it matters

Video reasoning has been bottlenecked by supervision — either expensive labels (planning traces) or expensive judges (large LLMs). Future-OPSD uses **temporal self-consistency** as the verifier, which scales linearly with raw video and transfers across domains.

The foresight score is the key ingredient: pure next-segment prediction without it over-weights trivial continuity, pulling the model toward frame-copying instead of reasoning.

---

## Gotchas & tricks

- **Short futures bias toward perception; long futures bias toward generic hallucination.** The right future-window length is empirical; the paper uses mid-range windows.
- **Foresight-score bootstrapping.** The score is trained jointly with the policy and can collapse (every frame weighted high, no signal) without regularization.
- **Static videos** (lectures, head-shots) provide no learnable future variation — filter them out of the training pool.

---

## Sources

- Paper: *Learning Video Foresight Reasoning from Unlabeled Videos via On-Policy Self-Distillation* — Hu, Ye, Jia, Liang, Fu, Tian, Zhao, Chen, 2026 — [arXiv:2611.26954](https://arxiv.org/abs/2611.26954)
