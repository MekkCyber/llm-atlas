# Sharpening Tax
*Depth — a diagnostic metric for post-training's cost in solution coverage.*

**TL;DR:** RL post-training mostly *sharpens* a base model's distribution — it raises pass@1 by concentrating probability on known-good solutions, at the cost of pass@K coverage. The **Sharpening Tax** quantifies that cost: it measures the loss of test-time scalability (how much pass@K grows with K) that a post-trained model pays versus its base. The Oh et al. (2026) study measures the tax across 14 base/post-trained pairs on three agentic benchmarks and shows the base model, with a lightweight harness, often beats its post-trained sibling at pass@K despite far worse pass@1.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../post-training/reasoning/length-penalty.md](../post-training/reasoning/length-penalty.md) · [math500.md](./math500.md)

---

## What it is

For a model $M$, prompt $p$, and sample budget $K$:

$$
\text{pass@K}(M, p) = 1 - \prod_{i=1}^{K} (1 - c_i), \quad c_i \in \{0, 1\} \text{ for the } i\text{-th rollout}
$$

The **slope** of pass@K vs K is the model's *test-time scalability*. The Sharpening Tax of a post-trained model $M_{\text{PT}}$ against its base $M_{\text{B}}$ is defined as the gap in that slope — equivalently, the K-asymptotic coverage gap:

$$
\text{SharpeningTax}(M_{\text{PT}} \mid M_{\text{B}}) \;=\; \lim_{K \to \infty} \big[\, \text{pass@K}(M_{\text{B}}) - \text{pass@K}(M_{\text{PT}}) \,\big]
$$

It is estimable from a handful of rollouts and correlates well with other diversity proxies.

## How it works

- Pick a benchmark where both pass@1 and pass@K are meaningful (SWE-bench Verified, Agentic Browsing, GPQA-Agentic).
- Run $K$ independent rollouts of base and post-trained models, score each.
- Fit per-prompt pass@K curves and aggregate across prompts.
- Report (a) pass@1 gap and (b) K-asymptotic pass@K gap — the tax.

Oh et al. introduce **Posterior-Tempered Group Sampling (PTGS)** as a mitigation: adapt the sampling temperature per prompt based on its estimated difficulty under a Bayesian prior, raising coverage on hard prompts without blowing up easy ones. Applied *during* RLVR training, PTGS lowers the tax and also improves pass@1 — the two aren't in strict opposition if the sampler is adaptive.

## Why it matters

- **Agentic deployments already pay for K.** Code agents, SWE agents, and parallel-rollout systems live at K ∈ [8, 128]. If RL kills coverage, the deployment setting silently loses on the exact axis it scales.
- **Reframes "post-training as sharpening"** as a measurable, moveable number — not a philosophical observation. "Our RL doesn't tax" is now a claim you can falsify.
- **Explains counterintuitive base-model wins.** A base model under a cheap inference harness beating the post-trained version at pass@K is no longer a surprise — the tax predicts it.

## Gotchas & tricks

- **Pass@K estimator variance** at small sample counts is high; use the unbiased estimator (Chen et al., 2021) and bootstrap confidence intervals.
- **Temperature confound.** Comparing base at T=0.7 against post-trained at T=0 understates the base; sweep temperature or report the Pareto frontier.
- **Not all RL is equal.** DPO, GRPO, and preference RM runs tax differently; measure yours before assuming the published numbers transfer.
- **Harness design matters.** A base model with CoT prompting + light tool wrapper is the right comparison; raw greedy decode understates the base.
- **Train-time mitigation beats test-time.** PTGS during RL pays a smaller tax than fixing the sampler at inference only.

## Sources

- Paper: *Sharpening Tax in Post-Training* — Oh et al., 2026 — [arXiv:2610.01509](https://arxiv.org/abs/2610.01509). Introduces the tax metric, measures it across 42 (model, benchmark) cases, and proposes PTGS.
- Paper: *Evaluating Large Language Models Trained on Code* — Chen et al., 2021 — pass@K estimator used throughout.
- Note: adjacent framing in *RL Does Not Teach New Capabilities* (Yue et al., 2024) and *Base Models Beat Aligned Models at Randomness* — complementary evidence.
