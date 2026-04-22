# Outcome Reward Model (ORM)

*Depth — a verifier trained from binary "is the final answer correct?" labels, used to rerank or reward whole solutions.*

**TL;DR:** Take a base LM, finetune a separate copy with a scalar head on top of whether its generated solutions reached the correct final answer (label = 1/0 from ground truth). The resulting model — the ORM — scores a full solution by how likely its final answer is correct. Used for **best-of-N reranking** (cheap test-time boost) and as a **reward signal for RL** (expert iteration, PPO/GRPO). Introduced by Cobbe et al. (2021) for GSM8K; formalized as the "outcome" side of the outcome-vs-process distinction by Uesato et al. (2022). The surprising empirical finding is that ORM training — binary final-answer labels only — produces a verifier that agrees well with human *step-level* judgments too, because the LM has to internalize what correct reasoning looks like in order to predict final-answer correctness.

**Prereqs:** [_post-training](../_post-training.md), [rlvr](../rlvr.md)
**Related:** [prm](prm.md), [_rewards](../_rewards.md), [rejection-sampling](../rejection-sampling.md), [long-cot-rl](long-cot-rl.md)

---

## What it is

An **outcome reward model** (ORM) is a learned function

```
ORM(q, o) → [0, 1]    # probability the final answer in solution o is correct
```

trained on a dataset of `(prompt q, solution o, correct_final_answer ∈ {0, 1})` triples. "Outcome" because the label comes from the **final outcome only** — not from judging the reasoning steps.

### Contrast with related things

| | Label source | What it scores | See |
| --- | --- | --- | --- |
| **ORM** | Final-answer correctness (rule check on ground truth) | Whole solution → one scalar | this page |
| **PRM** | Step-by-step human labels | Per step → scalars (aggregated) | [prm](prm.md) |
| **Rule verifier** | Ground truth, at inference | Whole solution → 0/1, no learning | [rlvr](../rlvr.md) |
| **Preference RM** | Human preferences between pairs | Whole response → scalar | [_post-training](../_post-training.md) |

A rule verifier and an ORM solve overlapping problems: both produce an "is this solution good?" scalar. The ORM **generalizes** — it can be used on prompts where you don't have ground truth at inference (held-out problems, rejection sampling from deployed models), because it's learned to *predict* correctness rather than checking it. The cost is that ORMs can be wrong; rule verifiers cannot.

---

## How it works

### Training (Cobbe 2021 recipe — the canonical one)

```
1. Generator G: finetune the base LM on (prompt, worked-solution) pairs
   (SFT; standard next-token loss).
2. For each training prompt q:
      sample K = 100 completions from G(q)
      label each: 1 if extracted_answer(o) == reference, else 0
3. Train ORM on the (q, o, label) triples.
```

**Architecture.** The ORM is the base LM with a **scalar prediction head**. In Cobbe's original recipe, the head predicts **at every token** (a token-level value function), not just at the end. At inference, the scalar at the final token is used as the solution's score. Training this way gives extra signal and helps the ORM generalize.

**Loss.** A combined objective:

```
L_ORM = L_verification + L_LM
```

The verification loss is per-token prediction of the solution's correctness label (the same 0/1 broadcast across every token of a given solution). The paper also keeps the standard language-modeling loss as an auxiliary. The LM auxiliary matters: verifiers trained only on the verification loss overfit fast.

**Important subtlety — per-token labels are uniform per solution.** Every token in a correct solution gets label 1; every token in a wrong solution gets label 0. This is a coarse signal — the ORM doesn't know *which* tokens were wrong, only that the solution ended up wrong. Despite this, the ORM **ends up assigning low scores around actual error locations** because those prefixes have low *empirical* probability of leading to a correct final answer.

### Inference usage — best-of-N

The simplest and historically first use:

```
at test time:
    sample K responses from G(q)
    return o_i with highest ORM(q, o_i)
```

In Cobbe 2021, this pushes a **6B model with a 6B ORM to roughly match a 175B finetuned baseline on GSM8K** — a ~30× effective model-size increase for the cost of K inference samples. The best-of-K curve rises sharply up to ~K=100, then plateaus; Cobbe notes that past ~400 samples, accuracy starts to *decrease* because the ORM occasionally scores a wrong solution above all right ones (reward-model error compounds with more candidates).

### RL usage — reward for policy optimization

The ORM can also be a **reward function for RL post-training**. Uesato et al. (2022) show this explicitly on GSM8K with **expert iteration** (sample K from the policy, keep the top-ORM sample, SFT on those):

```
for round:
    for each prompt q:
        sample K from π
        pick top_i = argmax_i ORM(q, o_i)
        SFT on (q, o_{top_i})
```

This is equivalent to **rejection-sampled SFT with the ORM as the filter** — see [rejection-sampling](../rejection-sampling.md). Uesato's final error rate (12.7% on GSM8K) came from SFT + ORM-RL (expert iteration with ORM).

Modern use: plug the ORM into [GRPO](../grpo.md) or PPO as the reward signal. The policy trains against ORM, with KL penalty to a reference model. Cheaper than a per-step PRM; more generalizable than a brittle rule verifier on ambiguous prompts.

### The surprising "ORM ≈ PRM" result

Uesato et al.'s headline empirical finding: on GSM8K, an ORM trained only on final-answer labels achieves **~85% agreement with per-step human PRM labels** — higher than with its own training labels (77%). In plain English: a reward model trained only on outcome information ends up implicitly learning to recognize bad reasoning steps, because those are causally upstream of wrong final answers.

The practical consequence: for math-style problems with clean ground-truth answers, ORMs capture most of the benefit of PRMs at a fraction of the labeling cost. This is the empirical backbone of why [RLVR](../rlvr.md) (pure outcome rewards) works so well — and why DeepSeek-R1 rejected PRM in favor of rule-based outcome rewards.

---

## Why it matters

- **Cheap labels.** Rule-based ground-truth checks produce ORM training labels for free, once you have a dataset with answer keys. No step-level annotation.
- **Test-time capability boost from the same base.** Best-of-N reranking with an ORM is the simplest non-trivial inference-time compute lever. Free if you were already sampling, cheap if not.
- **Drop-in reward for verifiable-domain RL.** Anywhere a rule verifier is too strict or not applicable (e.g., word problems where "42 apples" vs "42" matters; code problems where no test suite exists), a trained ORM fills the gap while keeping the outcome-only training signal.
- **Scales better than humans.** ORM quality grows with labeled data; labeled data is bottlenecked on solving problems, not on annotating them. Self-bootstrapping is possible: stronger generator → more correct solutions → better ORM training data → better ORM → better reranking → stronger generator.

---

## Gotchas & tricks

- **False positives from "right answer, wrong reasoning."** The ORM trusts the final answer, so a solution that reaches `42` by fluke is labeled correct. In practice this happens often enough to be a real confound in GSM8K training — Cobbe 2021 notes it explicitly. Mitigation: a [PRM](prm.md) on the same data; or use an ORM for filtering and a separate process-aware judge for reranking.
- **Reward hacking on learned ORMs.** If you use an ORM as the RL reward, the policy will find adversarial outputs that score high without being correct (e.g., certain prefix phrases the ORM treats as a "confidence signal"). Keep the KL penalty in RL; cap ORM training rounds; periodically evaluate against a rule verifier on held-out prompts. DeepSeek-R1 specifically avoids learned RMs during its reasoning RL for this reason.
- **ORM generalization falls off out-of-distribution.** An ORM trained on GSM8K-style problems won't reliably score AIME or Olympiad problems — the reasoning patterns differ. Use the ORM within its training distribution, or retrain.
- **Token-level vs solution-level scoring.** Cobbe's original ORM scores at every token; other implementations score only on the final token ("solution-level"). Token-level gives more signal and generalizes slightly better; solution-level is simpler and halves verifier-training compute. Either works; token-level is the stronger default.
- **LM auxiliary loss is load-bearing.** Without the co-trained language-modeling objective, Cobbe's ORMs overfit and lose generalization. If you're training an ORM from scratch, keep the LM loss as an auxiliary.
- **Temperature of the rollouts matters.** The ORM training data comes from sampling the generator. At temperature 0, every rollout is identical and you get no signal. Temperature around 0.7–1.0 is typical; Uesato used **T = 1.0** with K = 96.
- **The ORM-as-RL-reward vs ORM-as-reranker distinction.** Reranking (cheap, one-time at inference) is where ORMs shine. Using an ORM as a *dense RL reward* is where reward hacking typically appears; if you have the option to use a rule verifier for RL and save the ORM for reranking, do that.

---

## Sources

- Paper: *Training Verifiers to Solve Math Word Problems* — Cobbe et al., OpenAI, 2021, [arXiv 2110.14168](https://arxiv.org/abs/2110.14168) — introduces the verifier-style ORM, GSM8K dataset, and best-of-N reranking. Base models: GPT-3 family (primarily **6B and 175B**); K=100 rollouts per prompt; token-level verifier with joint verification + LM loss.
- Paper: *Solving Math Word Problems with Process- and Outcome-Based Feedback* — Uesato et al., DeepMind, 2022, [arXiv 2211.14275](https://arxiv.org/abs/2211.14275) — names the ORM/PRM distinction, reports the "ORM agrees with PRM labels ~85%" result, runs expert-iteration RL with both ORM and PRM rewards on Chinchilla-70B ("Our Base-70B"). Best GSM8K final-answer error: **12.7%** with SFT + ORM-RL.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — discusses outcome vs process rewards and explicitly avoids learned neural RMs (including ORMs) during reasoning RL to prevent reward hacking.
- Related: [orm](orm.md) is the RL-reward / reranker artifact; [rlvr](../rlvr.md) is the pure-rule-based version of the same idea with deterministic verifiers instead of a learned ORM.
