# Process Reward Model (PRM)

*Depth — a verifier that scores each step of a reasoning trace, not just the final answer.*

**TL;DR:** Train a model to predict, for every step of a chain-of-thought, whether that step is correct given the prior steps. Collecting labels is the hard part: **Lightman et al. (2023)** hired humans to annotate 800,000 step-level judgments (the **PRM800K** dataset); **Math-Shepherd (Wang et al., 2024)** replaced humans with Monte-Carlo rollouts that estimate step quality as the fraction of completions from that prefix reaching the correct final answer. Used for **best-of-N reranking** (aggregate per-step scores into a solution score) and as a **dense RL reward** (per-step advantage shaping). Conceptually attractive — local credit assignment, catches reasoning errors even when the final answer is right — but fragile: **DeepSeek-R1 tried PRMs and abandoned them**, citing step-definition ambiguity, label cost/noise, and reward hacking.

**Prereqs:** [_post-training](../_post-training.md), [orm](orm.md)
**Related:** [rlvr](../rlvr.md), [_rewards](../_rewards.md), [mcts](mcts.md), [long-cot-rl](long-cot-rl.md)

---

## What it is

A **process reward model** (PRM) is a function

```
PRM(q, o_1..i) → score_i    for i = 1..S          (S = number of steps)
```

scoring each intermediate step of a reasoning trace. Contrast with an [ORM](orm.md), which returns one scalar for the whole solution.

Three things PRMs can be used for:

1. **Best-of-N reranking.** Sample K full solutions. Compute per-step scores for each. Aggregate (min, product, mean) into a solution score. Rerank by solution score. This was Lightman's main use.
2. **RL reward shaping.** Use per-step scores as dense rewards during policy optimization — each step's advantage depends on its own PRM score rather than a single terminal reward broadcast across all tokens. Math-Shepherd does this with PPO.
3. **Search guidance.** The PRM serves as a value function for tree search (MCTS, beam search). See [mcts](mcts.md) — ReST-MCTS* uses a PRM this way.

---

## How it works

### Where step labels come from

The hard part of PRMs isn't the architecture or the loss — it's **labeling**. Three main recipes:

**(a) Human step annotation (Lightman 2023 — "Let's Verify Step by Step").** Humans read each step and mark it positive / negative / neutral. Expensive but high-quality. PRM800K contains **800k step-level labels** over ~75k solutions to ~12k problems — at the time, the largest step-level dataset ever collected.

**(b) Monte-Carlo rollouts (Math-Shepherd, Wang 2024).** No humans. For each step `s_i` in a solution, sample `N` continuations from `s_i`. The step's quality is the **fraction of continuations that reach the correct final answer**:

```
MC(s_i) = (1/N) · Σ_j  1[answer(rollout_j) = reference]
```

Two label variants:
- **Hard Estimation (HE):** label = 1 if any rollout succeeds from this prefix, else 0.
- **Soft Estimation (SE):** label = the MC fraction itself (real number in [0,1]).

(c) **Human + model hybrid (Uesato 2022).** Humans label a smaller seed set; a learned PRM bootstraps labels on additional data.

Monte-Carlo labeling eats compute, but removes the human bottleneck and scales — this is why Math-Shepherd and its descendants are the practical default now.

### PRM architecture

The PRM is a base LM with a **classification head** that predicts correctness at a designated boundary token (end of each step). Training is standard:

```
L_PRM = - Σ_i [ y_i · log PRM(s_{1..i}) + (1-y_i) · log(1 - PRM(s_{1..i})) ]
```

where `y_i ∈ {0, 1}` (HE labels) or `y_i ∈ [0, 1]` (SE labels) and `PRM(s_{1..i})` is the sigmoid output after step `i`. A single forward pass over the full solution yields all per-step scores.

### Aggregating step scores into a solution score

For reranking, aggregate `score_1..score_S` into one number per solution:

| Aggregation | Interpretation | Used by |
| --- | --- | --- |
| **Product** | Probability every step is correct | **Lightman 2023** (main) |
| **Minimum** | Worst-step score (pessimistic) | **Math-Shepherd** (for verification) |
| **Average** | Mean step confidence | Ablations |
| **Last step** | Confidence at the final step | Sometimes; similar to an ORM |

The product and the minimum are the two most principled choices. Lightman's default is the product (equivalent to a chain of independent correctness predictions); Math-Shepherd uses the minimum because it's more robust to spurious "over-confident" steps.

### Using a PRM as an RL reward

Math-Shepherd runs **step-by-step PPO** where each step gets a dense reward from the PRM. Instead of a single terminal reward broadcast to all tokens, the advantage at token `t` is driven by the PRM's score at the step `t` belongs to. This gives finer credit assignment: a policy that writes a correct first step and a wrong second step gets rewarded for the first and penalized for the second, separately.

This is the PRM's theoretical advantage over an ORM: **local credit assignment**. In practice the gains are modest — Math-Shepherd reports Mistral-7B going from 77.9% → 84.1% on GSM8K with step-by-step PPO, vs gains from the same model with outcome-only RL that are in a similar range.

### Headline empirical results

**Lightman 2023** (finetuned from **GPT-4**, best-of-N reranking at **N=1860**):
- MATH test, representative subset: **78% problems solved** via PRM reranking of GPT-4 samples — best number on MATH at the time.
- PRM reranker outperforms ORM reranker by a consistent margin on hard problems.

**Math-Shepherd (Mistral-7B):**
- GSM8K: **77.9% → 84.1%** with step-by-step PPO; **89.1%** with the PRM as a verifier (rerank) on top.
- MATH: **28.6% → 33.0%** with PPO; **43.5%** with verification.

**Uesato 2022 (Chinchilla-70B, GSM8K):** PRM-RL and ORM-RL essentially tied — PRM-RL marginally worse on both final-answer and trace error. This is the result that prompted the "ORM is surprisingly competitive" takeaway.

---

## Why it matters

- **Local credit assignment.** A PRM tells you *which* step went wrong, not just that the solution failed. In principle this should dramatically improve policy gradients over sparse outcome rewards. In practice the effect is smaller than hoped — but still real, especially on long reasoning chains.
- **Catches reasoning errors when the final answer is right.** ORMs trust final answers; PRMs don't. For high-stakes applications where you care about the reasoning (not just the number), PRMs are strictly better signals.
- **Foundation for MCTS with LLMs.** Tree search over reasoning needs a per-step value estimate. PRMs provide one. See [mcts](mcts.md).
- **Pushes the test-time-compute frontier.** PRM-reranked best-of-N at large K was (briefly) the SOTA recipe on MATH — Lightman 2023's 78% was a big jump at release. Superseded by long-CoT RL (R1-class models), but the PRM+search family continues in parallel.

---

## Gotchas & tricks

- **What counts as a "step" is ill-defined.** For math word problems you can split on newlines or numbered lists. For general reasoning — "explain the tradeoffs of X" — there is no clean step boundary. This was **DeepSeek-R1's first stated reason** for abandoning PRMs: *"it is challenging to explicitly define a fine-grain step in general reasoning."*
- **Labeling is the cost center, not training.** PRM800K took human effort measured in person-years. Monte-Carlo labels (Math-Shepherd) replace humans with compute, but the compute is substantial: `N` continuations per step per training problem, where `N` often ≥ 8. For a large training set this becomes a real budget item.
- **Reward hacking of learned PRMs is real.** The policy can discover prefix patterns the PRM rewards, even when the steps aren't actually correct. **DeepSeek-R1's third stated reason** for rejecting PRMs: *"a model-based PRM inevitably leads to reward hacking."* Mitigation: KL penalty, frequent eval against rule verifiers, cap PRM-as-reward training epochs.
- **MC-labeled PRMs have label noise.** A step can be "correct" in some continuations and "wrong" in others. HE labels smooth this by taking any-success; SE labels keep the noise but use it as a gradient signal. Pick carefully — SE is higher variance, HE is biased.
- **Product aggregation can underflow on long solutions.** 30 steps with p = 0.9 each gives 0.04. Use log-probabilities for numerical stability; prefer min-aggregation on very long chains.
- **PRMs shine on reranking, struggle on RL.** The cleanest empirical signal is that PRMs as rerankers reliably beat ORMs on hard math. PRMs as RL rewards are, at best, tied with ORM-RL and frequently unstable. If you only do one, do reranking.
- **The "ORM is surprisingly competitive" result (Uesato 2022)** matters for planning: an ORM trained on pure outcome labels ends up agreeing with human step labels ~85% of the time. Much of the benefit of PRMs is latent in ORMs trained on enough data. A rule-verifier-driven ORM or — better — pure [RLVR](../rlvr.md) often matches PRM reranking at a fraction of the cost.
- **Not all "PRMs" in recent papers are the same thing.** Some "PRMs" are per-token critics used as RL baselines (closer to a value network); others are per-step binary classifiers (this page's focus). The labeling procedure and aggregation matter more than the name — always check.

---

## Sources

- Paper: *Solving Math Word Problems with Process- and Outcome-Based Feedback* — Uesato et al., DeepMind, 2022, [arXiv 2211.14275](https://arxiv.org/abs/2211.14275) — formalizes the process/outcome distinction; shows on GSM8K that ORM-RL and PRM-RL are roughly tied.
- Paper: *Let's Verify Step by Step* — Lightman et al., OpenAI, 2023, [arXiv 2305.20050](https://arxiv.org/abs/2305.20050) — PRM800K (**800,000 step labels**), PRM reranking of GPT-4 solutions, **78% on MATH representative subset**. The canonical human-labeled-PRM paper.
- Paper: *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations* — Wang et al., 2024, [arXiv 2312.08935](https://arxiv.org/abs/2312.08935) — automated step labels via Monte-Carlo rollouts (HE / SE), step-by-step PPO, min-aggregation for reranking. Mistral-7B: GSM8K **84.1% / 89.1%**, MATH **33.0% / 43.5%**.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — explicitly tried PRMs in reasoning RL and abandoned them; see §"Unsuccessful Attempts." Cites Uesato, Lightman, and Math-Shepherd as references.
- Dataset: [PRM800K](https://github.com/openai/prm800k) — the step-labeled dataset from Lightman 2023.
