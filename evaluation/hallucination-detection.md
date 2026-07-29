# Hallucination Detection over Reasoning Traces (ReDe)
*Depth — a step-level denoising front-end that filters noisy reasoning steps before feeding traces into any downstream hallucination detector.*

**TL;DR:** Long chain-of-thought traces from large reasoning models carry useful signal about whether the final answer is hallucinated — but they also contain irrelevant and repetitive steps that drown the signal. **ReDe** uses the model's own **final-answer attention** as automatic supervision to learn a step-level embedding space in which noisy steps are separable, then filters them out before running any hallucination detector. Consistent improvements over confidence-based scores and naive embedding filters across reasoning benchmarks.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../evaluation/README.md](../evaluation/README.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

Reasoning-model hallucination detectors face a specific problem: as CoT length grows, the fraction of steps that carry answer-relevant signal shrinks, and the rest is noise (repeated exploration, irrelevant tangents, dead-end backtracks). Confidence-based scores get confused by low-confidence *exploration* steps that don't actually matter for the final answer. ReDe is a denoising front-end that removes noisy steps before any downstream detector runs.

## How it works

1. **Automatic supervision from final-answer attention.** For a trace with a final answer, use the model's attention weights *from the final-answer tokens back to earlier trace steps* as a soft relevance label — steps that the answer actually attended to are informative; steps ignored by the answer are candidates for noise.
2. **Learn a step-level embedding space** shaped by that supervision, in which relevant steps cluster distinctly from noisy ones.
3. **Filter.** At detection time, embed each step, drop steps flagged as noisy (irrelevant or repetitive), and hand the *cleaned* trace to a downstream hallucination detector — any detector; ReDe is a plug-in front-end.

Two noise categories are named explicitly: **irrelevant** (steps unrelated to the eventual answer) and **repetitive** (steps that duplicate earlier reasoning without adding information).

## Why it matters

As long-CoT and reasoning-RL models become default, downstream monitoring has to work over increasingly long, noisy traces. A cheap, model-owned denoising step in front of any hallucination detector cleanly separates two problems — *what to look at* and *how to score it* — that were previously entangled. Adjacent to CoT monitoring safety work but aimed at truthfulness rather than misuse.

## Gotchas & tricks

- Final-answer-attention supervision only works when the final answer exists and is not itself corrupted — for wildly wrong answers the attention signal degrades.
- The step-level embedding needs to be trained per model family; attention patterns are not portable.
- Repetitive-step detection is sensitive to paraphrase-level repeats — semantic embeddings help; hashing does not.
- Combines with *any* downstream detector, but the marginal gain shrinks when the detector already has a strong internal filtering step.

## Sources

- Paper: *Reasoning Denoiser: Denoising Reasoning Traces for Hallucination Detection in Large Reasoning Models* — Fang et al., 2026 — [arXiv:2607.22098](https://arxiv.org/abs/2607.22098)
