# Reward Types for LLM RL

*Taxonomy — where the scalar reward signal comes from in RL post-training, and how to choose.*

**TL;DR:** Every RL post-training recipe needs a way to score a response. The options fall into five families — **rule-based / verifiable**, **outcome reward model (ORM)**, **process reward model (PRM)**, **preference reward model**, and **shaping rewards** — and they differ in three axes: *correctness* (how often the signal is right), *cost* (compute per score), and *hackability* (how easy it is for the policy to game). The modern reasoning stack prefers the left end: use a rule verifier when you can, a learned RM only when you must.

**Related taxonomies:** [_rl](_rl.md) · [_post-training](_post-training.md)
**Depth files covered here:** [rlvr](rlvr.md) · [orm](reasoning/orm.md) · [prm](reasoning/prm.md)

---

## The problem

RL optimizes the policy toward higher reward. The *identity* of the reward function sets the ceiling. Three failure modes every reward choice has to dodge:

1. **Reward is wrong** — the signal disagrees with what you actually want. Learned RMs trained on finite data will do this on OOD prompts.
2. **Reward is hackable** — the policy finds adversarial outputs that score high but aren't good. Every learned RM is hackable to some extent; rule verifiers usually aren't.
3. **Reward is too sparse or too noisy for the RL update to converge.** If every rollout scores 0, or the variance of scores is uninformative, the policy gradient is zero or dominated by noise.

Picking the reward type is mostly about trading these three off for your task.

---

## The shared pattern

Every LLM RL reward has this shape:

```
R(prompt, response) → scalar   (possibly vector, if combined)
```

Variants differ in **who produces the scalar**:

| Family | Producer | Cost per score | Label source |
| --- | --- | --- | --- |
| Rule-based verifier | Deterministic code | ~free | Ground-truth-aware code (math answer match, test-pass, format check) |
| [ORM](reasoning/orm.md) | Learned LM with a scalar head | 1 RM forward pass | Final-answer correctness labels |
| [PRM](reasoning/prm.md) | Learned LM, per-step scores aggregated | 1 RM forward pass | Per-step labels (human or MC rollouts) |
| Preference RM (RLHF-style) | Learned LM with a scalar head | 1 RM forward pass | Human pairwise preferences |
| Shaping / format / language | Deterministic code or tiny classifier | ~free | Composed into the full reward |

Most production recipes use **more than one** — e.g., rule verifier + format reward + language-consistency reward (DeepSeek-R1 Stage 2), or rule verifier for reasoning + preference RM for general helpfulness (R1 Stage 4). The reward is usually just the **sum** of the components, weighted.

---

## Variants

### 1. Rule-based (verifiable) rewards — [RLVR](rlvr.md)

**Signal:** deterministic code checks. Math: does the extracted answer match the reference? Code: do the unit tests pass? Format: is the output the required structure?

**Pros.** Cheap. Perfectly aligned with ground truth — no learned reward-model slop. Basically unhackable at the reward layer (the policy can't "fool" a regex). Strong scale properties: throw more prompts and more RL compute at it, the model improves.

**Cons.** Only works for domains where correctness can be mechanically checked. Useless for "is this response kind," "does this answer the question well," etc.

**Who uses it.** DeepSeek-R1, DeepSeekMath, Tülu 3's RLVR stage, every math/code reasoning RL paper in 2024–2025.

See: [rlvr](rlvr.md), [long-cot-rl](reasoning/long-cot-rl.md).

### 2. Outcome Reward Model ([ORM](reasoning/orm.md))

**Signal:** a learned LM with a scalar head, trained on `(prompt, solution, correct_final_answer ∈ {0,1})` labels — where the label comes from a rule verifier at training time but the ORM generalizes to prompts where no verifier is available.

**Pros.** Generalizes beyond the domain where you have rule verifiers. Cheaper than PRMs (no step-level labels). Surprisingly strong — Uesato 2022 showed ORM-RL ties PRM-RL on GSM8K.

**Cons.** Hackable (learned RM). Label noise from false positives (solution reaches right answer by luck). Narrow generalization if trained on one task distribution.

**Who uses it.** Cobbe 2021 (GSM8K verifier, reranking). Uesato 2022 (expert iteration with ORM). Many modern pipelines keep an ORM available for rerank-style inference boosts.

See: [orm](reasoning/orm.md).

### 3. Process Reward Model ([PRM](reasoning/prm.md))

**Signal:** per-step correctness scores from a learned LM trained on step-level labels. Aggregated (product, min, mean) into a solution score, or used directly as dense per-step RL reward.

**Pros.** Local credit assignment — you know *which* step was bad. Catches "right answer, wrong reasoning." Strong for reranking; pairs naturally with MCTS/search.

**Cons.** Step definitions are ill-posed for general reasoning. Label cost (humans) or compute cost (Monte-Carlo) is high. Reward hacking is worse than ORMs because the reward is dense. **DeepSeek-R1 tried this and abandoned it** for exactly these reasons.

**Who uses it.** Lightman 2023 (reranking); Math-Shepherd (PPO with step rewards); ReST-MCTS*. Less common in frontier reasoning RL.

See: [prm](reasoning/prm.md).

### 4. Preference Reward Model (classical RLHF)

**Signal:** learned LM trained on human pairwise preferences (A > B for prompt q). Scores whole responses by a scalar that is monotonic in preference.

**Pros.** Works for open-ended tasks where "correct" isn't defined (helpfulness, tone, creative writing). The only reward type that captures human aesthetic judgments at scale.

**Cons.** The most hackable of all reward types. Trained on a narrow slice of human judgment; OOD behavior is often weird. Expensive to train, maintain, and serve.

**Who uses it.** InstructGPT, classical RLHF, Claude, GPT-4 post-training. DeepSeek-R1 Stage 4 uses it only for helpfulness on general prompts (scored on the summary), while reasoning prompts use rule verifiers.

See: [_post-training](_post-training.md) (preference-optimization section).

### 5. Shaping / auxiliary rewards

**Signal:** small cheap terms added to the main reward to shape secondary behaviors.

Common examples:

| Shaping term | What it does | Where used |
| --- | --- | --- |
| Format reward | `+1` if response matches a required schema (XML tags, JSON) | R1-Zero, RLVR |
| Language-consistency reward | Proportion of target-language words in CoT | R1 Stage 2 |
| Length penalty | Reward decreases with excessive length | Various; careful |
| Repetition penalty | Penalize n-gram repeats | Various |

**Pros.** Cheap, targeted, composable.

**Cons.** Every shaping reward is a lever for reward hacking. Stack too many and the policy optimizes for shaping at the expense of correctness. Direct-sum composition (R1-style) sacrifices a small amount of headline performance for the shaping benefit — **a real, named tradeoff** that you see recur in every paper that adds them.

---

## How to choose

The decision tree for picking a reward type, for an RL post-training run:

```
Is the task verifiable (math, code, format, structured output)?
│
├── Yes → rule-based verifier + RLVR. Use this by default.
│         Add format / language shaping rewards for UX polish.
│
└── No → Do you have human preference data?
         │
         ├── Yes → preference RM + PPO  OR  DPO (preferred — no rollouts).
         │         For reasoning-with-fuzzy-correctness: train an ORM from rule-verifier
         │         labels on a sibling task, then use it where verifiers don't apply.
         │
         └── No → either collect preferences, or fall back to SFT with
                   rejection sampling (use a judge LLM as the filter).
```

Additional rules of thumb:

- **Pair RLVR with preference RM in stages.** Separate *capability* (RLVR on reasoning) from *style* (preference RM on general). R1's 4-stage pipeline does this; so do Tülu 3 and most modern recipes.
- **If you can use a rule verifier, use it.** Don't train a learned RM "just in case." Learned RMs introduce alignment risk and compute cost that rule verifiers don't.
- **PRM is a reranker, not an RL reward.** The literature mostly agrees by 2025: PRMs are useful at inference for rerank/search; using them as dense RL rewards is brittle.
- **Shaping rewards should be small.** A 5–10% contribution relative to the main reward is typical. If the shaping reward is load-bearing, the main reward isn't doing its job.

---

## Adjacent but distinct

- **DPO-style preference training** skips the reward function entirely — the policy update is expressed directly in terms of preference log-ratios. Same signal type (preferences) as a preference RM; different implementation (no RM to train/serve). See [_post-training](_post-training.md) for the preference section.
- **Constitutional AI / RLAIF** replaces human preference labels with LLM-judge labels, then trains a preference RM (or does DPO) on those. Orthogonal to this taxonomy — it's about label generation, not reward identity.
- **Model-as-judge at inference** (Best-of-N with a judge LLM) reuses reward-model ideas outside RL. It's a test-time compute lever, not a training signal.

---

## Sources

- Paper: *Training language models to follow instructions with human feedback* — Ouyang et al., 2022 — canonical preference-RM RLHF.
- Paper: *Training Verifiers to Solve Math Word Problems* — Cobbe et al., 2021 — origin of the outcome-style verifier / ORM.
- Paper: *Solving Math Word Problems with Process- and Outcome-Based Feedback* — Uesato et al., 2022 — names the process/outcome distinction and shows they're near-tied.
- Paper: *Let's Verify Step by Step* — Lightman et al., 2023 — large-scale PRM study (PRM800K).
- Paper: *Math-Shepherd* — Wang et al., 2024 — automated step labels; PRM as PPO reward.
- Paper: *Tülu 3* — AI2, 2024 — introduces the "RLVR" name and the rule-based reward recipe.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — explicit choice of rule-based rewards over neural RMs; tried and rejected PRMs for RL.

---

## Conventions

- **Filename:** `_rewards.md` (leading underscore, taxonomy).
- **Folder placement:** `post-training/` root, same level as [_rl.md](_rl.md) and [_post-training.md](_post-training.md). Links into `reasoning/` for ORM and PRM.
- **Scope:** the *signal* side of RL post-training, in contrast to [_rl.md](_rl.md) which covers the *optimization* side.
