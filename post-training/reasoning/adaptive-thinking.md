# Adaptive Thinking (Difficulty-Aware Length Control)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Hybrid reasoning models overthink easy prompts and underthink hard ones. **Adaptive thinking** replaces a global length penalty with **per-instance difficulty signals** — precomputed accuracy and token-usage statistics for each training example — that regulate reasoning depth so the model spends its tokens where they buy accuracy.

**Prereqs:** [length-penalty.md](length-penalty.md), [long-cot-rl.md](long-cot-rl.md)
**Related:** [long2short.md](long2short.md), [../grpo.md](../grpo.md)

---

## What it is

A hybrid reasoning model has two modes — direct response and long chain-of-thought — separated by a `<think>` marker or similar. The naive control knob is a **length penalty** in the reward, but a single penalty either over-compresses easy problems (silently losing accuracy on hard ones) or under-compresses hard problems (wasting tokens on easy ones).

Adaptive thinking (introduced as **Instance-level Difficulty-Aware Control**, or IDAC) treats *difficulty* as an instance-level property: for each training prompt, compute reference statistics — accuracy at N samples, token usage at N samples — from a strong baseline, then feed those statistics into the training and inference objectives as a per-example budget.

## How it works

1. **Reference statistics.** For each training prompt, run a reference model (or the pre-adaptation policy) N times. Record per-prompt accuracy and mean tokens-per-response.
2. **Difficulty label.** Bucket prompts by accuracy (high/medium/low). Optionally quantize tokens-per-response to a per-bucket target.
3. **Length shaping.** During SFT / RL, weight the length term by the per-prompt target — easy prompts get a tighter budget, hard prompts get more room. During inference, the model uses a learned or heuristic gate on whether to enter thinking mode.
4. **Loss.** The RL / SFT loss is standard except the length regularizer is now `f(target_t) · length` rather than a global constant.

The strong-baseline reference is what makes IDAC different from earlier "self-paced" schemes — the model doesn't have to *discover* difficulty online; it's handed a difficulty prior.

## Why it matters

- **Better accuracy-vs-tokens Pareto.** Fixed-depth reasoning wastes tokens on trivial prompts and clips reasoning on hard ones; per-instance budgets move the frontier up.
- **Deployment lever.** Serving cost of hybrid reasoning models is dominated by long-CoT prompts; adaptive thinking reduces the fraction of prompts that pay the long-CoT cost.
- Reported gains: cleaner AIME24 and AIME25 accuracy/token tradeoffs than fixed-length baselines.

## Gotchas & tricks

- **Reference statistics need refresh.** If the reference model is far from the policy, the difficulty labels drift out of calibration; recompute periodically.
- **Difficulty is not uniform per benchmark.** GSM8K "easy" and AIME "easy" are different problems — never share reference statistics across dataset domains.
- **Budget leakage.** If the model can peek at the difficulty label at inference (e.g., through a system-prompt hint), it will over-fit to the label rather than the problem. Use budgets only in the reward, not as an inference-time input.

## Sources

- "When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models" — Shim et al., 2026 — [arXiv:2609.19671](https://arxiv.org/abs/2609.19671).
