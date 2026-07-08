# LLM-as-a-Verifier (Continuous-Score Verification)

*Depth — treat verification as a scaling axis: continuous scores, repeated evaluation, criteria decomposition.*

**TL;DR:** Standard LLM-judge pipelines prompt the model to output a discrete score (e.g., 1–10) and use the sampled token. Discrete scores tie frequently on hard candidate pairs. LLM-as-a-Verifier (2026) instead takes the **expectation over the score-token logit distribution** — $E[\text{score}] = \sum_k p(k) \cdot k$ — producing a continuous score with near-zero tie rate. Once scores are continuous, verification scales along three orthogonal axes: finer score grids, repeated evaluations, and criteria decomposition. No additional training; the same pretrained model provides a dense enough signal to serve as a reward for RL post-training. SOTA on Terminal-Bench V2 (86.5%), SWE-Bench Verified (78.2%), RoboRewardBench (87.4%), MedAgentBench (73.3%).

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md) · [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md) · [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md) · [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

Prompt the LLM with a candidate solution and a rubric; force it to answer with a score token in a discrete vocabulary $\{k_1, \ldots, k_M\}$. Instead of sampling one token, read the softmax over the score vocabulary directly:

$$
\hat s = \sum_{k \in \text{score vocab}} p_\theta(k \mid \text{prompt}) \cdot k
$$

The output is a continuous number in the score range. Preference between two candidates $a$ and $b$ is the sign of $\hat s_a - \hat s_b$; probability of preferring $a$ is a monotone function of that difference.

## How it works

Three scaling axes, empirically each independently useful:

1. **Score granularity.** Increase $M$ (10 → 100 → 1000 score tokens). Continuous scores get sharper; ties on close candidates vanish. Beyond a point, granularity plateaus because the underlying logits are noisier than the added bins.
2. **Repeated evaluation.** Query the verifier $N$ times with permuted candidates / rubric orderings; average continuous scores. Reduces variance; the classic ensemble-of-judges trick, but on continuous scores it composes multiplicatively with granularity.
3. **Criteria decomposition.** Split the rubric into orthogonal criteria (correctness, style, safety, …), score each independently, sum with learned or hand-tuned weights. Reduces per-criterion complexity for the verifier; more calibrated aggregation.

Ranking: build a preference matrix from $\hat s$; apply a cost-efficient tournament to pick top-K without $O(N^2)$ pairwise queries.

## Why it matters

- **Fixes the tie-rate failure mode.** Discrete LLM-judge scores tie frequently on hard candidate pairs, discarding useful signal. Continuous scores don't.
- **Verification as a compute-scaling knob.** Pretraining scaling, post-training scaling, and test-time scaling are the standard three; this paper argues verification is a fourth, with clean scaling curves along granularity/repetition/decomposition.
- **Dense reward for RL.** Continuous scores plug into GRPO/SAC as a smooth reward, improving sample efficiency vs sparse verifier signals — a real answer for domains where rule-based verifiers don't exist (medical, agentic, robotics).
- **Progress signals.** The paper shows continuous scores also work as a proxy for task-progress monitoring — a shipped extension for Claude Code and Codex leverages this for agent introspection.

## Gotchas & tricks

- **Score token structure.** The score vocabulary must be adjacent tokens the model can address individually (e.g., digits + decimal). Multi-token scores (e.g., "eighty-seven") break the expectation trick.
- **Rubric matters.** Continuous scores don't fix an unclear rubric. Decomposition helps because each sub-criterion is easier for the verifier to reason about.
- **Calibration is domain-specific.** A score of 7/10 means different things across domains; leave the raw scores for RL and normalize at aggregation for ranking.
- **Bias to the prompt template.** Small template changes shift the expected score. Fix the template within a run.
- **Repetition budget.** Diminishing returns after ~8 repetitions per candidate; put more compute into decomposition once repetition plateaus.
- **Not a replacement for rule verifiers.** When a rule verifier exists (math answer match, unit test pass), it dominates. LLM-as-Verifier is the answer where rules don't work.

## Sources

- Paper: *LLM-as-a-Verifier: A General-Purpose Verification Framework* — Kwok, Li, Atreya, Liu, Jiang, Finn, Pavone, Stoica, Mirhoseini, 2026 — introduces continuous-score verification and the three scaling axes.
- Related: *Generative Verifiers: Reward Modeling as Next-Token Prediction* — Zhang et al., 2024 — generative reward-model predecessor.
- Related: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — Zheng et al., 2023 — the discrete-score LLM-judge baseline this paper improves.
