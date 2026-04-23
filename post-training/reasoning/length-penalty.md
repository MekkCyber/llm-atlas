# Length Penalty (Reward Shaping for Long-CoT RL)
*Depth — the asymmetric reward-shaping term that stops long-CoT RL from overthinking.*

**TL;DR:** In long-CoT RL, response length grows unboundedly — the model learns that thinking longer helps, and there's nothing stopping it from padding. Kimi k1.5 adds a **group-relative linear length reward** that interpolates between **+0.5** (shortest) and **−0.5** (longest) within each group of `k` rollouts to the same problem. The key design choice: **the positive half is only awarded to correct responses**. Incorrect responses get `min(0, λ)` — penalized for being long, never rewarded for being short. Stops reward hacking ("give up quickly to save tokens") while preserving the token-efficiency signal. Introduced as a **warm-up schedule**: disabled for an initial RL phase, then applied with constant weight.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [rlvr](../rlvr.md), [_rewards](../_rewards.md)
**Related:** [online-policy-mirror-descent](online-policy-mirror-descent.md) · [long2short](long2short.md) · [kimi-k1-5 case study](../../case-studies/kimi-k1-5.md)

---

## What it is

A **shaping reward** added to the main outcome reward during RL. Introduced in Kimi k1.5 (Moonshot AI, 2025, Sec. 2.3.3) as a specific fix for the **overthinking** phenomenon — the empirical observation that under pure outcome-reward long-CoT RL, response length grows uncontrolled: from hundreds of tokens early, to thousands mid-training, to unboundedly long (thousands-to-tens-of-thousands) late.

Overthinking has two costs:
- **Inference cost.** Test-time compute scales with CoT length.
- **Training cost.** Longer rollouts mean fewer rollouts per unit time, slowing RL.

A length penalty solves both if it's shaped correctly. Naive length penalties (subtract `α · length` from the reward) tend to cause the model to *give up* on hard problems — emit short wrong answers, get penalized zero for length, pay only the outcome-reward cost. k1.5's design prevents that failure mode through an **asymmetric reward**.

---

## How it works

### The formula

For `k` sampled responses to problem `x`, let `len(i)` be the token length of response `i`, `min_len = min_i len(i)`, `max_len = max_i len(i)`.

If `max_len == min_len`: length reward = 0 for all responses (no signal when all responses are the same length).

Otherwise, compute a **group-relative linear score**:

```
λ(i) = 0.5  −  ( len(i) − min_len ) / ( max_len − min_len )
```

Then apply the **asymmetric gating**:

```
len_reward(i)  =  {  λ(i)         if r(x, y_i, y*) = 1    (correct)
                  {  min(0, λ(i)) if r(x, y_i, y*) = 0    (incorrect)
```

### Read as a diagram

Within a group of `k` rollouts ordered by length (shortest → longest):

```
Length:                    min_len ─────────────────────── max_len
λ(i):                        +0.5  ─────────────────────── −0.5

Correct responses:          +0.5   ─────── 0 ─────── −0.5
                            (shortest                  longest
                             correct                   correct
                             gets + bonus)             gets − penalty)

Incorrect responses:         0     ─────── 0 ─────── −0.5
                            (shortest                  longest
                             incorrect                 incorrect
                             no bonus —                gets − penalty)
                             floored at 0)
```

### Why the asymmetry

If both correct *and* incorrect responses could earn positive length bonus:
- A short incorrect response would score `outcome = 0, length = +0.5, total = +0.5`.
- A long correct response would score `outcome = 1, length = −0.5, total = +0.5`.

The policy is indifferent — which means on hard prompts (where correctness is low), it learns to give up quickly for the length bonus. The `min(0, λ)` floor on incorrect responses removes this — incorrect short gets 0 (no bonus), incorrect long gets negative (penalty). Now the only way to earn positive reward is to be **correct**, and among correct responses, short is preferred.

This is a small line of code with a real reward-hacking defense inside it. Worth understanding per-detail.

### Combined with the outcome reward

The length reward is **added to the outcome reward** with a weighting coefficient:

```
total_reward(i)  =  r_outcome(x, y_i, y*)  +  w · len_reward(i)
```

The paper does **not disclose `w`**. Ballpark: outcome reward is `{0, 1}`, length reward is `[-0.5, +0.5]` — so `w ≈ 1` gives similar magnitude; `w < 1` keeps outcome dominant.

### Warm-up schedule

Sec. 2.3.3 (last paragraph) specifies a two-phase schedule:

1. **Phase 1**: disable length penalty. Run standard policy optimization (outcome reward only) for an initial period. Let the model **first discover** that longer responses help.
2. **Phase 2**: enable length penalty at a **constant weight** for the rest of training.

Duration of phase 1 is not disclosed. The purpose of the warm-up: applying length penalty too early suppresses the emergence of long-CoT capability itself (the model never discovers that thinking longer helps). Once the capability is established, the penalty can pressure the model toward token-efficient expressions of the same reasoning.

### Why group-relative, not absolute

The reward compares *within a group of responses to the same prompt*. Hard prompts will have longer responses in general; easy prompts shorter. A global threshold (e.g., "penalize if len > 4000") would penalize long responses on hard problems (where length is earned) and fail to penalize long responses on easy problems (where length is padding). Group-relative normalization handles this automatically: each prompt is judged on *its own length distribution*, and the penalty pushes toward the shortest correct response for that particular prompt.

---

## Why it matters

- **Solves overthinking cleanly.** Length collapse via naive penalties (global `α · length`) is a common failure mode. The asymmetric-gated group-relative form is a specific, reusable recipe that avoids it.
- **Directly enables [long2short](long2short.md).** Kimi's long2short-RL variant reuses this length reward with a tighter max-length cap, producing a short-CoT model at the same accuracy ceiling. The length penalty is the core primitive.
- **Named alignment-cost instance.** Adding the penalty costs a small amount of peak-accuracy in exchange for dramatically shorter responses. Worth reporting as a tradeoff — part of the *"shaping rewards have a named cost"* pattern seen across R1's language-consistency reward and others.
- **Composable with other rewards.** Additive to the outcome reward, with a single coefficient. Easy to add / remove for ablation.

---

## Gotchas & tricks

- **Needs the warm-up phase.** Applying length penalty from step 0 suppresses long-CoT emergence entirely. Train first with outcome reward only; turn on length penalty after long-CoT is established.
- **The `min(0, λ)` floor is load-bearing.** If you simplify to "just add `λ` for all responses", you get the give-up-early failure mode. Preserve the asymmetry.
- **Group-relative means `k` matters.** With `k = 2`, the length reward is just `{+0.5, −0.5}` — no nuance. With `k = 16`, it's a proper gradient. Small `k` makes the signal noisy.
- **Max_len == min_len edge case.** The formula divides by `(max_len − min_len)`. When all responses are the same length (rare but possible), the paper explicitly sets length reward = 0. Implementations must handle this.
- **Interacts with repetition.** A policy that learns to get longer by repeating itself (degenerate long output) will pay the length penalty but also inflate `max_len` for other responses in the group, which flattens the signal. Combine with repetition penalty / early termination (see the partial-rollouts mechanism: [partial-rollouts](../../systems/partial-rollouts.md)).
- **Coefficient `w` is undisclosed.** The paper says "added with a weighting parameter" and stops there. Treat as a tunable ~1.0 starting point; probably `w ∈ [0.1, 1.0]`.
- **Length in tokens, not words.** Tokenization differences across models mean "5000 tokens" isn't a fixed real-world length. Compare within-tokenizer.
- **Doesn't fix all length issues.** The penalty reduces average length but doesn't make short-CoT competitive with long-CoT at peak performance — for that, use long2short distillation after RL.
- **Don't generalize to non-verifiable tasks naively.** The asymmetry depends on a clean correctness signal. For RLHF with preference rewards (no binary correct/incorrect), the `min(0, λ)` floor doesn't map cleanly. Needs adaptation.

---

## Sources

- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI (Kimi Team), 2025, arXiv 2501.12599 — introduces the length-penalty formula (Sec. 2.3.3) and the warm-up schedule.
- Related context: *DeepSeek-R1* — DeepSeek-AI, 2025 — notes that length grows autonomously under long-CoT RL without length reward (Sec. 4 of the R1 paper), but does not impose a length penalty; R1 accepts the cost. The contrast frames why k1.5's length reward is load-bearing for token efficiency.
