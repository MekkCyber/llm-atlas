# Long-CoT RL — Reasoning from RL Alone

*Depth — pure-RL training from a base model that grows long, reflective chain-of-thought without any SFT demonstrations.*

**TL;DR:** Start from a pretrained base (no SFT, no CoT examples), run [GRPO](../grpo.md) with a [verifiable reward](../rlvr.md), and the model **autonomously grows longer and longer reasoning traces** — hundreds of tokens at first, thousands later — and **spontaneously develops reflection, verification, and backtracking**. The length and the reasoning patterns are not rewarded directly; they emerge because they help the model score on hard problems. Demonstrated at scale by **DeepSeek-R1-Zero** (2025): RL from `DeepSeek-V3-Base` drove AIME 2024 pass@1 from **15.6% → 77.9%** without a single human CoT label.

**Prereqs:** [grpo](../grpo.md), [rlvr](../rlvr.md)
**Related:** [orm](orm.md), [prm](prm.md), [_rewards](../_rewards.md), [deepseek-r1 case study](../../case-studies/deepseek-r1.md)

---

## What it is

The classical recipe for teaching an LLM to reason is:

```
Base → SFT on human-written CoT demonstrations → RL on verifiable tasks
```

The SFT step is where the model learns *what a reasoning trace looks like*. Without it, people expected RL would not know where to start: the base model wouldn't produce well-structured reasoning at all, verifier rewards would be near-zero, and the policy gradient would have nothing to latch onto.

**Long-CoT RL** is the empirical demonstration that this intuition is wrong. Given a strong enough base model (trained on enough code, math, and web text that it has latent reasoning capacity) and a pure-RL loop with verifiable rewards, the model:

1. Discovers on its own that **longer responses score better** on hard problems,
2. Grows its outputs from a few hundred to several thousand tokens over training,
3. **Spontaneously** invents reflection (*"wait, let me check that"*), self-verification, and alternative-strategy exploration,
4. Reaches reasoning-benchmark parity with SFT-bootstrapped reasoners.

The category name "long-CoT RL" is post-hoc — the technique is just RL with a verifier — but it's useful to name the phenomenon because it reframes what RL does for LLMs: not only does it sharpen known capabilities, it can **elicit new behavioral structures** that weren't in the SFT data (because there was no SFT data).

---

## How it works

### The recipe

```
while not converged:
    prompt_batch = sample_math_or_code_problems()
    for each prompt q:
        {o_1, ..., o_G} ~ π_θ(· | q)           # G rollouts, full responses
        r_i = verifier(q, o_i)                 # 1 if correct, 0 if not
        A_i = (r_i - mean(r)) / std(r)         # group-relative advantage
    policy.update_GRPO(advantages, KL_to_ref)
```

No value network, no preference model, no learned reward. Two signals only: **accuracy reward** (did the final answer verify?) and a tiny **format reward** (did the model wrap its reasoning in `<think>...</think>` and its answer in `<answer>...</answer>`?).

### The prompt template

R1-Zero used this exact conversation template during RL — no in-context examples, no CoT demonstrations, just structural scaffolding:

> *A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within `<think></think>` and `<answer></answer>` tags, respectively.*

The model is told *that* it should think, not *how*. Everything else is emergent.

### The emergent trajectory

Across tens of thousands of RL steps, three things happen in sequence:

**1. Response length grows.** The model starts with typical base-model-ish short outputs. It quickly learns that on hard problems, more tokens of work before the answer correlate with higher reward. Average response length rises from hundreds to several thousand tokens — with no length reward, only verifier reward. The extra tokens buy it more chances to be right.

**2. Reasoning structure appears.** Early in training, the "thinking" is just longer blobs of computation. Mid-training, the model starts **structuring** it: numbered steps, sub-problems, intermediate checks. None of this is prompted or rewarded.

**3. Reflection emerges.** Late in training, the model begins saying things like *"Wait, let me re-examine step 3..."* or *"Actually, that's not right. Let me try a different approach."* It re-evaluates partial solutions and switches strategies mid-chain. The DeepSeek-R1 paper calls this the **"aha moment"** — not because the model suddenly becomes conscious, but because the researchers observed it spontaneously developing a problem-solving heuristic (self-revision) that no one put there. The paper's own phrasing: *"This moment is not only an 'aha moment' for the model but also for the researchers observing its behavior."*

### Why SFT isn't strictly required

The common worry — "the base model won't produce any correct solutions, so the gradient is zero" — is empirically wrong for strong bases. A few reasons:

- **Latent capability from pretraining.** Modern bases (DeepSeek-V3, Llama 3, Qwen 2.5) have seen enormous amounts of math textbooks, code, and worked examples. They already *know* how to solve some AIME-level problems if sampled aggressively. The base-model pass@K for reasonable K is non-trivial; that's all RL needs to bootstrap.
- **Group-relative advantages handle sparsity.** With `G = 16` rollouts per prompt, even a 5–10% success rate gives a strong within-group ranking signal. [GRPO](../grpo.md) normalizes by the group's own mean and std, so even sparse rewards produce informative advantages.
- **Verifier-free tokens are cheap.** Each RL step requires only prompt sampling + verifier check + GRPO update. No human labels, no critic, no RM inference. You can afford to train for a long time on a curriculum of hard problems.

### R1-Zero vs R1

R1-Zero is the pure-long-CoT-RL result. R1 is the productized model built on top: DeepSeek found that R1-Zero had two user-facing problems — **poor readability** (the CoT is stream-of-consciousness) and **language mixing** (Chinese and English interleave within the same trace). So R1 bolts on a **cold-start SFT** (~thousands of clean CoT examples) before the RL, and a **language-consistency reward** during RL (proportion of target-language words in the CoT). Both fixes sacrifice a little benchmark performance for a much cleaner user experience.

The point worth internalizing: **long-CoT RL from base works**; SFT and shaping rewards are there for polish, not capability.

---

## Why it matters

- **It decouples reasoning from human CoT annotation.** You don't need an army of labelers writing worked solutions. You need a verifier and a strong base.
- **It's the clearest demonstration of emergence via RL at LLM scale.** Prior emergence stories (ICL, CoT-from-scratch) were emergent-from-pretraining. Long-CoT RL is emergence-from-RL: patterns appear in the policy that were not present in the base and were not in any training signal.
- **It pushes test-time compute up autonomously.** The model decides for itself to "think more." Length is a learned strategy, not a decoded hyperparameter. This is the cleanest instance of *RL discovers test-time compute scaling*.
- **It is the training recipe behind the frontier reasoner class** (o1, R1, Kimi k1.5, and descendants). Everyone post-R1 does some form of RL-for-reasoning with verifiable rewards; the "long-CoT" framing comes from this paper.

---

## Gotchas & tricks

- **Needs a strong base model.** Long-CoT RL from a small or under-trained base doesn't work — the paper's own control experiment trained Qwen-32B-Base with >10k RL steps and the resulting `R1-Zero-Qwen-32B` **underperformed** the 32B *distilled* from R1's traces. Capability ceiling is set by the base. Small models: prefer distillation.
- **Reward hacking is still possible.** The model can learn to write the correct final answer with garbage reasoning, or to invent a fake check-step that happens to always output "correct." Keep the format reward simple, watch eval-vs-train divergence, and add behavioral evals (not just the trained verifier).
- **Readability is not free.** Pure R1-Zero traces are hard to read — mixed languages, run-on thinking, no summary. If you want a product, expect a second pass: cold-start SFT for readability priors, plus a **language-consistency reward** (e.g., proportion of target-language words in the CoT) during RL. This costs a small amount of benchmark performance in exchange for a much better user-facing trace.
- **Length collapse at convergence.** Past a point, continued training can *shrink* response length again as the policy tightens. Monitor length and pass@1 jointly — the peak is not necessarily at the longest traces.
- **Rule-based rewards only.** Long-CoT RL depends on a verifier. If your task has no programmatic check (most chat), this isn't applicable. Use [preference-RM RLHF](../_post-training.md) or [DPO](../_post-training.md) instead. See [_rewards](../_rewards.md) for the full landscape.
- **Don't expect emergence for "free."** The compute is real. DeepSeek ran R1-Zero on the 671B/37B-active DeepSeek-V3 MoE for thousands of RL steps. Emergent reflection is cheap *per step* (no critic, no RM) but expensive *in total*.
- **PRM and MCTS don't help here.** The DeepSeek-R1 paper tried both and abandoned both. See [prm](prm.md) and [mcts](mcts.md) for why. Long-CoT RL with an outcome-only verifier outperformed process-level reward shaping and search-augmented rollouts.

---

## Sources

- Paper: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — DeepSeek-AI, 2025, arXiv 2501.12948, *Nature* 645, 633–638 — introduces R1-Zero, the canonical long-CoT RL demonstration.
- Paper: *DeepSeekMath* — Shao et al., 2024 — introduces [GRPO](../grpo.md), the policy-optimization algorithm used for R1-Zero.
- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI, 2025, arXiv 2501.12599 — contemporaneous long-CoT RL result using [online policy mirror descent](online-policy-mirror-descent.md) (an ℓ₂-regression sibling of GRPO) with an asymmetric [length-penalty](length-penalty.md) reward, [partial-rollouts](../../systems/partial-rollouts.md) infrastructure, and an explicit [long2short](long2short.md) distillation phase. See the [kimi-k1-5 case study](../../case-studies/kimi-k1-5.md) for the full pipeline and an R1↔k1.5 contrast table.
