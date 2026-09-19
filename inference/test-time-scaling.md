# Test-Time Scaling
*Depth — spending more inference compute per prompt to raise accuracy, and the schedule that decides the true cost.*

**TL;DR:** Test-time scaling covers all inference-side techniques that spend more compute per prompt to raise answer quality: best-of-N, majority voting, verifier trees, hedged decoding. The reported knob is usually the candidate count $N$. Kashaniyan & Jannesari (2026) show that $N$ is *underspecified* — the **schedule** of how those $N$ candidates are executed (one batched call vs. many serial calls) dominates the systems cost. Fix $N=8$: 8 serial calls burn 4.64–4.86× the GPU-device energy and 5.77–6.12× the P95 latency of one 8-candidate batched call on A100.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

An umbrella term for inference recipes that produce and combine multiple candidate answers rather than sampling once:

- **Best-of-N (BoN).** Sample $N$ candidates, pick the top-scored one under a verifier or reward model.
- **Majority voting.** Sample $N$, vote — pick the most common answer.
- **Tree search / MCTS.** Explore a search tree of partial responses with a value model.
- **Hedged decoding.** Speculatively decode multiple continuations, commit only one.

All of them convert *more compute* into *better answers* at the same base model. All of them report their quality-vs-cost tradeoff as a function of $N$ (or search-tree budget).

---

## How it works

A generic test-time-scaling call:

```
prompt ─► generate N candidates ─► score / aggregate ─► return best
```

The "generate N candidates" step has a hidden knob: **the generation schedule**. For the same $N$, you can:

- $1 \times N$ — one call, batch of $N$ candidates.
- $a \times b$ with $a \cdot b = N$ — $a$ sequential calls of batch $b$.
- $N \times 1$ — $N$ fully serial calls of batch $1$.

Same output, wildly different serving cost. On A100 with Phi-3-mini and Qwen2.5-1.5B, holding $N = 8$:

| Schedule | Gross GPU-device energy | P95 latency |
| --- | --- | --- |
| $1 \times 8$ | 1× | 1× |
| $8 \times 1$ | 4.64–4.86× | 5.77–6.12× |

The pattern reproduces across three A100 nodes per model and in short-output SciQ/V100 experiments. When candidates are independent and memory allows, fewer generation calls with larger batches are strictly more efficient.

---

## Why it matters

- **Every reasoning paper reports Pass@k or BoN@N.** Without the schedule, "Pass@8" is a range spanning nearly an order of magnitude in real cost.
- **Serving-side implications.** Load balancers and rollout workers that fan out per-candidate calls unknowingly pay 5× energy for the same $N$. Batched-first is the default that saves cost.
- **Extends to RL rollouts.** GRPO / long-CoT-RL rollout workers face the same choice for $k$ rollouts per prompt. Same math applies.
- **Reporting norm proposal.** The paper's ask — report $N$ *and* schedule *and* GPU-level systems metrics — should become standard for test-time-scaling evaluations.

---

## Gotchas & tricks

- **Memory is the batching ceiling.** $1 \times N$ wins only when $N$ candidates fit in a single batch. Above that, some serialization is forced — but keep batches as large as fits.
- **Non-independent candidates change the picture.** In verifier-guided tree search, later candidates depend on earlier ones, so full batching isn't possible. Report schedule anyway.
- **Latency vs. throughput tradeoff.** P95 latency drops with batching (the paper's finding); mean latency can rise slightly because a batch pays for the slowest sample. Pick the metric that matches your SLA.
- **Comparing across hardware needs care.** The 5× number is A100-specific; H100/B200 have different memory-bandwidth ratios.
- **Speculative / hedged decoding is a separate axis.** It reduces per-candidate cost but doesn't change the $a \times b$ schedule story.

---

## Sources

- Paper: *Sample Count Is Not Enough: Candidate-Generation Strategy Shapes the Energy and Performance of LLM Test-Time Scaling* — Kashaniyan & Jannesari, 2026 (Iowa State U.) — [arXiv:2609.19499](https://arxiv.org/abs/2609.19499).
- Prior art: best-of-N as a scaling law: Snell et al., 2024.
