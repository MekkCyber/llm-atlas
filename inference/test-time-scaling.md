# Test-Time Scaling (TTS)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Spend more compute at inference — typically by generating N candidate answers per prompt and picking the best — to raise accuracy without touching model weights. The naïve knob is *candidate count N*; the underappreciated knob is the **batching schedule** (one N-wide call vs several serial calls) which changes GPU energy and P95 latency by 4–6× at fixed accuracy. A fair TTS number requires both.

**Prereqs:** [README.md](README.md), [../post-training/reasoning/README.md](../post-training/reasoning/README.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md), [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)

---

## What it is

Test-time scaling is any inference-time strategy that trades extra tokens for higher expected quality on a fixed model:

- **Best-of-N (BoN)** — sample N responses, select by a verifier / reward model / self-consistency vote.
- **Self-consistency** — sample N chains of thought, majority-vote the final answer.
- **Tree / MCTS search** — expand a search tree over partial responses using a process reward model.
- **Repeated refinement** — iteratively critique and rewrite.

The universal knob is compute: more samples or deeper search generally lifts accuracy, up to a plateau that depends on the base model's coverage of the answer space.

## How it works

A BoN pipeline for N=8 candidates on a math prompt:

1. Sample 8 completions from the policy. This can be **one call at batch=8** (single prefill, single KV cache reused across the batch) or **8 sequential calls at batch=1** (repeated prefill of the shared prompt, no KV reuse across calls).
2. Score each completion (verifier, reward model, or vote).
3. Return the top-scoring completion.

Step 1's schedule dominates system-level cost. On A100 with Phi-3-mini and Qwen2.5-1.5B on GSM8K, 8 serial calls consume **4.64–4.86× more GPU-device energy** and hit **5.77–6.12× the P95 latency** of a single batched call — accuracy identical.

## Why it matters

- **A cheap accuracy dial** for deployments that can afford extra tokens.
- **Model-agnostic** — TTS composes with any base model without retraining.
- **Reporting hygiene** — TTS papers commonly compare methods at fixed N without disclosing schedule. That is a factor-of-6 gap in inference cost that can flip the tradeoff.

## Gotchas & tricks

- **Report schedule, energy, and latency.** Sample count alone under-describes cost. Recent work advocates including GPU-level metrics as a first-class TTS reporting requirement.
- **KV-cache reuse.** A single wide-batch call reuses the prefilled KV across all N samples; serial calls do not. For long prompts this is the dominant win.
- **Verifier quality gates gains.** A weak verifier can make N > 4 actively harmful — it selects confident-but-wrong candidates. Pair BoN with a process reward model or an outcome verifier trained for the task.
- **Diminishing returns** — accuracy typically plateaus by N ≈ 16–32 for math and code; beyond that, sampling more just costs more.

## Sources

- "Sample Count Is Not Enough: Candidate-Generation Strategy Shapes the Energy and Performance of LLM Test-Time Scaling" — Kashaniyan & Jannesari, 2026 — [arXiv:2609.19499](https://arxiv.org/abs/2609.19499).
- "Self-Consistency Improves Chain of Thought Reasoning in Language Models" — Wang et al., 2022 — foundational self-consistency reference.
