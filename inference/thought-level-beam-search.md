# Thought-Level Beam Search
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Test-time reasoning cast as a **constrained compute-allocation problem over partial trajectories**. Instead of running $N$ independent traces (parallel sampling) or pruning without branching (pure subtractive), periodically prune weak partial trajectories and immediately branch from strong prefixes. A lightweight scorer that reads hidden states plays the value-function role. Introduced by Gambit (Princeton/MIT/Nvidia, 2026).

**Prereqs:** [../post-training/reasoning/README.md](../post-training/reasoning/README.md), [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)
**Related:** [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Test-time compute is the main knob for reasoning models. The dominant baselines occupy two extremes:

- **Parallel sampling ($N$ independent traces).** Simple, embarrassingly parallel, but treats every trace as independent — hardware bloats with growing KV cache, no cross-trace reallocation.
- **Subtractive pruning.** Kill low-scoring traces mid-generation. Reclaims memory but starves the accelerators (fewer active sequences) and doesn't sufficiently shift the output distribution — you get less compute on the *good* traces.

Thought-level beam search sits between: prune the bad, branch from the good, keep hardware saturated.

## How it works

Sequence generation is broken into **thoughts** (semantic chunks, e.g. sentences or reasoning steps). At the end of each thought:

1. **Score each active trajectory** using a lightweight scorer that probes the model's hidden states at the current cursor. No separate large PRM required; the scorer is small and reads what's already in the forward pass.
2. **Prune** trajectories in the bottom fraction of the score distribution.
3. **Branch** from top-scoring prefixes: fork the prefix's KV cache into new active trajectories, immediately using the memory that pruning just freed.
4. **Maintain constant width.** Pruning + branching happens in one step, so the number of active trajectories is kept near the hardware sweet-spot — high utilization, no idle capacity.

The key operational detail is **KV-cache-aware branching**: forking from a shared prefix reuses the prefix's cache pages (via paged attention), so branching is nearly free in memory. That's what lets Gambit dynamically re-allocate compute without paying to re-prefill.

## Why it matters

- **Better use of a fixed hardware budget.** Under identical hardware constraints, Gambit reports **+6.7% absolute accuracy on HMMT-24** and **+3.3% on AIME-25** over pruning baselines.
- **Higher throughput.** >2× throughput on trace completion vs standard parallel sampling.
- **Fewer tokens for the same answer.** Up to **68.5% total-token reduction** vs standard parallel sampling — a large real-world serving cost win.
- **Middle-ground design point.** Fits between MCTS (heavy per-step planning) and parallel sampling (no planning), giving practitioners a serving-friendly beam-search option for reasoning.

## Gotchas & tricks

- **The hidden-state scorer is the key ingredient.** A weak scorer makes pruning noisy and branching wasted. The paper reports the scorer works because it reads late-layer hidden states where task-relevant features are cleanest.
- **Thought boundary detection matters.** Fixed-token intervals over-prune long thoughts and under-prune short ones. Semantic boundaries (period, "let me reconsider", answer marker) work better in practice.
- **Branching factor vs width.** Higher branching from the top prefix concentrates compute but loses diversity — the surviving set can converge on one wrong path. Keep at least a few low-branching survivors as diversity insurance.
- **Not a replacement for PRMs when scoring is expensive.** If your task genuinely needs a heavy process reward model to judge partial reasoning quality (e.g., theorem proving with formal checks), Gambit's cheap scorer isn't enough — treat the scorer as a fast pre-filter and run the PRM at the final step.
- **KV cache implementation must support forking.** vLLM / SGLang paged-attention systems do; naive contiguous-cache systems don't and will re-prefill on every branch, killing the win.

## Sources

- Paper: *Thought-Level Beam Search for Reasoning* — Yang, Luo, Zhao, Dao, Netravali — Princeton, MIT, Nvidia, 2026 — [arXiv:2608.08020](https://arxiv.org/abs/2608.08020).
- Related: [mcts.md](../post-training/reasoning/mcts.md) — MCTS as a heavier tree-search cousin; [prm.md](../post-training/reasoning/prm.md) — the PRM family whose lighter cousin plays the value role here.
