# Evaluator-Verifier Reward (EVR)
*Depth — propose-then-verify decomposition for MLLM-based rewards on subjective outputs.*

**TL;DR:** Using a multimodal LLM as a zero-shot judge on complex generation tasks (multi-reference image editing, video edits, agent trajectories) has a known tension: **long-form reasoning hallucinates** and **short-form judgment is shallow**. EVR splits evaluation into two stages per criterion — an **Evaluator** proposes multiple candidate hypotheses, then a **Verifier** grounds each candidate in concrete evidence and accepts or rejects it. The accept-count becomes a dense, calibrated reward that avoids both failure modes.

**Prereqs:** [_rewards.md](./_rewards.md), [cot-reward-model.md](./cot-reward-model.md)
**Related:** [rlsvr.md](./rlsvr.md)

---

## What it is

A compositional reward-model pattern for scenarios where the reward signal must judge a rich, multi-dimensional output. The core idea: replace the single-model *"assess this output"* query with a two-model *"propose claims, then verify each claim against evidence"* pipeline.

The pattern is most cleanly demonstrated in EVR for multi-reference image editing, where an editor must maintain consistency across several reference images. But it generalizes to any RL fine-tuning loop that leans on a large model for subjective scoring.

## How it works

For a task with `k` evaluation criteria (e.g., "identity preservation," "lighting consistency," "scene harmony"):

1. **Decompose.** Split the reward computation into `k` per-criterion computations.
2. **Evaluator (propose).** For each criterion, an MLLM Evaluator generates **multiple candidate hypotheses** about whether the output satisfies the criterion — each a specific, grounded claim (e.g., "the subject's face in output matches reference A because …").
3. **Verifier (ground).** A separate MLLM Verifier checks each hypothesis against the actual visual evidence and returns accept or reject. The Verifier's job is *narrow* — evaluate a specific claim — which is where MLLMs are strongest.
4. **Aggregate.** Reward = weighted sum of verified accept-counts across criteria.
5. **Feed into RL.** Standard reward interface — EVR fine-tunes off-the-shelf editors (e.g., Qwen-Image-Edit) with no architectural changes.

## Why it matters

- Fixes the long-form-hallucination vs. short-form-shallowness tradeoff by splitting each into a task that plays to MLLM strengths.
- Rewards are fine-grained (per criterion) and grounded (per hypothesis), so gradient direction is more informative than a single scalar preference judgment.
- Generalizable pattern: applies wherever a large model is the least-bad option for scoring — agent trajectory quality, complex code edits, video edits.
- Empirically, moves Qwen-Image-Edit to match/surpass NanoBanana on multi-reference consistency + harmony metrics.

## Gotchas & tricks

- Verifier prompts must demand evidence citation. Without that, the Verifier degenerates into the Evaluator and both errors return.
- Criterion decomposition is domain-specific and needs care — poorly chosen criteria give noisy rewards.
- Same MLLM can serve as both Evaluator and Verifier if prompted differently, but the more independent they are (different models, different context) the more robust the reward.

## Sources

- Paper: *Evaluation-Verification Reward for Consistent Multi-Reference Image Editing* — Zhang et al. (XJTU · Amap Alibaba · SJTU), 2026 — [arXiv:2607.29025](https://arxiv.org/abs/2607.29025)
