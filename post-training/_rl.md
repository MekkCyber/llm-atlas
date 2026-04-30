# Reinforcement Learning for LLMs

*Taxonomy — the RL concepts that show up across LLM post-training: what a policy is, why we need a baseline, how PPO clips, how KL pulls back, and where each trick comes from.*

**TL;DR:** For LLMs, RL means: the model generates a response, a **reward** is assigned, and the policy is nudged to make high-reward responses more likely. The entire modern stack — **PPO, GRPO, DPO, RLVR, RLHF** — is variations on one equation (the policy gradient) with different answers to four questions: *what's the reward, how do we estimate the advantage, how do we keep the policy close to the starting point, and do we need a separate value network?* This taxonomy walks through those questions and the concepts each technique needs.

**Related taxonomies:** [_post-training](_post-training.md) · [_rewards](_rewards.md)
**Depth files covered here:** [ppo](ppo.md) · [grpo](grpo.md) · [dpo](dpo.md) · [online-policy-mirror-descent](reasoning/online-policy-mirror-descent.md) · [rlvr](rlvr.md) · [long-cot-rl](reasoning/long-cot-rl.md) · [prm](reasoning/prm.md) · [orm](reasoning/orm.md)

---

## The problem

Supervised learning teaches the model to **imitate** a reference completion. It's bounded by the quality of those references: the model will never exceed its teacher, and for open-ended tasks (write a helpful answer, solve a hard math problem) there isn't one "correct" reference anyway.

RL sidesteps this by training on a **scalar signal of quality** instead of a target token sequence. If we can *score* a completion (rule-based check, preference RM, code test), we don't need to specify what the completion should be — the model searches for completions that score well. For reasoning in particular, this is how you get performance that exceeds any human-written solution in the training data.

The shape of RL for LLMs is unusual in two ways:

1. **One-step "episodes."** The model generates a response, gets a reward, and the episode is over. There's no environment state evolving over many steps. (Strictly speaking each token is a step, but the reward is terminal.)
2. **Enormous action space.** The "action" at each step is a token from a 32k–256k vocabulary. This makes value learning and exploration different from classical RL.

Most of the LLM-specific tricks (GRPO's group baseline, reference-model KL penalty, DPO's closed-form policy update) are adaptations to these two realities.

---

## The shared pattern — the policy gradient

All policy-gradient RL is the same basic equation. Let the model be a stochastic policy `π_θ(a|s)` over actions `a` given state `s`. The **expected return** is:

```
J(θ) = E_{τ ~ π_θ} [ R(τ) ]
```

where a trajectory `τ = (s_0, a_0, s_1, a_1, ...)` is whatever the agent does, and `R(τ)` is the total reward along that trajectory. We want to increase `J(θ)`.

The **policy gradient theorem** says:

```
∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) · A_t ]
```

Read as English: **the gradient is a weighted sum of log-probabilities of the actions you actually took, weighted by how "good" each action was** (the *advantage* `A_t`).

Every RL algorithm in this taxonomy is *this equation* with different choices for:

- **What counts as an action/state.** For LLMs: state = prompt + tokens so far, action = next token. The episode ends at EOS.
- **How `A_t` is estimated.** Raw reward? Reward minus a baseline? A learned value function? Group-relative normalization?
- **How `π_θ` is stepped.** Vanilla gradient step? Clipped ratio (PPO)? Closed-form update (DPO)?
- **How we stop the policy from drifting.** KL penalty to a reference? Trust region? Nothing?

Keeping this one equation in mind makes all the variants much less scary.

---

## Core concepts, in order

### 1. State, action, policy

For an LLM during RL:

```
state s_t  = prompt + tokens generated so far (q, o_{<t})
action a_t = the next token o_t
policy π_θ(o_t | q, o_{<t}) = softmax over vocabulary, from the LLM
```

A **trajectory** `(q, o_1, o_2, …, o_T)` is a complete response. The trajectory probability under `π_θ` is:

```
π_θ(o | q) = Π_t π_θ(o_t | q, o_{<t})
```

and `log π_θ(o|q) = Σ_t log π_θ(o_t | q, o_{<t})`. This is the same object the model outputs during pretraining (autoregressive likelihood) — we just call it a "policy" now because we're going to optimize it.

### 2. Reward

For LLM RL, the reward is almost always **terminal** — given at the end of the response, not per token:

```
R(q, o) = single scalar assigned to the full response
```

Different RL recipes disagree on where this scalar comes from:

- **Rule-based verifier.** `R = 1` if the answer is correct, else `0`. Used by [RLVR](rlvr.md). See the [rewards taxonomy](_rewards.md).
- **Learned preference reward model (RM).** `R = RM(q, o)`, a neural net trained on human preferences. Used by classical RLHF.
- **Process reward model.** Per-step scores aggregated (min, product, average). See [prm](reasoning/prm.md).
- **Outcome reward model.** Learned classifier on "is this solution correct." See [orm](reasoning/orm.md).
- **Composite.** Sum of the above, e.g., `R = r_accuracy + r_format + r_language` (R1's Stage 2 does this).

The single scalar is broadcast to all tokens in the response when computing the policy gradient (unless the reward is explicitly per-step, as with a PRM used as reward shaping rather than reranker).

### 3. Baseline and advantage

The raw return `R(τ)` has **high variance**: some prompts are hard, some easy, and the policy gradient can be dominated by which prompts came up in the batch. The fix is to subtract a **baseline** `b(s)`:

```
A_t = R - b(s_t)
```

Any `b` that doesn't depend on the action keeps the gradient unbiased but reduces variance. Common choices:

| Baseline | Who uses it | Cost |
| --- | --- | --- |
| Zero (no baseline) | Vanilla REINFORCE | Noisy |
| Per-prompt running mean | Old tricks | Cheap |
| **Learned value network `V(s)`** | PPO, classical RLHF | Train a second ~LLM-sized model |
| **Group mean over K rollouts** | **[GRPO](grpo.md)**, modern default for reasoning | Free once you're already sampling K per prompt |

GRPO's choice — skip the value network and use the empirical mean of K rollouts from the same prompt — is the biggest practical speedup over PPO for LLMs, and why it's the default for verifiable-reward RL.

When the reward is sparse and binary (RLVR), the group std is also useful:

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

This **normalizes** the advantage so the update magnitude is independent of how hard the prompt is. Hard prompts with low reward variance give small updates; easy prompts with high variance give big ones. Self-regulating.

### 4. The KL penalty and the reference model

If you just maximize `R`, the policy will drift toward whatever the reward function happens to reward — often degenerate. Two canonical failure modes:

- **Reward hacking** — model discovers a trick the verifier accepts but humans wouldn't. Especially bad for learned RMs.
- **Capability loss** — model's general language quality degrades because the RL reward doesn't care about fluency or honesty.

The fix used everywhere is a **KL penalty to a reference model** `π_ref` (usually the SFT checkpoint, or the base model if there's no SFT):

```
L = -E[R(τ)] + β · E[ KL( π_θ || π_ref ) ]
```

With `β ∈ [0.001, 0.1]`. Read as English: *"maximize reward, but stay close to the reference policy."* The KL is computed per token:

```
KL( π_θ || π_ref ) = Σ_t Σ_v  π_θ(v|...) · log( π_θ(v|...) / π_ref(v|...) )
```

In practice it's estimated by a single-sample estimator:

```
KL ≈ log π_θ(o_t|...) - log π_ref(o_t|...)
```

The reference model is **frozen**. It's the "anchor" that keeps your RL run from turning the policy into a reward-hacking monster.

### 5. The PPO clip

Policy gradient says *make the log-probability of good actions go up*. Naively taking a big gradient step can push the policy so far that the `π_θ_old` used to sample the trajectory is no longer representative of `π_θ` — the update becomes off-policy and unstable. PPO's fix is the **clipped ratio**:

```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)

L_PPO = E [ min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t ) ]
```

with `ε = 0.2` standard. English: *"increase the probability of good actions — but clip the increase so you can't push it more than 20% per step."* The `min(…)` formulation means: for positive advantage, clip the upside; for negative advantage, clip the downside. Pessimistic either way.

PPO is the algorithm behind classical RLHF ([InstructGPT](https://arxiv.org/abs/2203.02155)) and remains the default when you have a learned value network. For LLMs with a sparse verifier, [GRPO](grpo.md) replaces the value network with a group-mean baseline but keeps the PPO clip.

### 6. Full GRPO objective, annotated

Combining everything above, the [GRPO](grpo.md) objective is:

```
J_GRPO(θ) = E_{q, {o_i}~π_old} [
    (1/G) Σ_i min( r_i(θ) · A_i,
                   clip(r_i(θ), 1-ε, 1+ε) · A_i )
    - β · KL( π_θ || π_ref )
]
```

with

```
r_i(θ) = π_θ(o_i | q) / π_θ_old(o_i | q)
A_i    = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

Pieces, matched to concepts 1–5 above: `π_θ/π_θ_old` ratio (PPO clip); `A_i` group-relative (GRPO baseline); `r_i` scalar terminal reward from a [rule-based verifier](rlvr.md) or learned RM; `β · KL` pulls back toward `π_ref`.

### 7. Online vs offline RL (and where DPO sits)

**Online RL** samples fresh rollouts from the current policy every step. PPO, GRPO, and all the "reasoning RL" papers are online.

**Offline RL** works from a fixed dataset of `(prompt, response, reward)` tuples, updating the policy without new samples. Much cheaper per step; no need for a rollout infrastructure.

**DPO** (Direct Preference Optimization) is the offline cousin of RLHF. It starts from the same RLHF objective — maximize preference-RM reward with a KL penalty to `π_ref` — and shows that for **pairwise preference data** the optimal policy has a closed form:

```
π*(o|q) ∝ π_ref(o|q) · exp( R(q,o) / β )
```

Rearranging to express `R` as a log-ratio of policies, you can eliminate `R` entirely and train directly on preferences. No reward model, no rollouts. See the [preference section of _post-training](_post-training.md).

DPO is the modern default when your training signal is human preferences. PPO/GRPO are the defaults when your signal is a scalar reward you can evaluate at train time.

---

## Variants used across this atlas

| Technique | Reward source | Advantage estimator | Clip/trust region | Needs value net? | When it wins |
| --- | --- | --- | --- | --- | --- |
| **REINFORCE** | Any scalar | Raw return | None | No | Toy/historical |
| **[PPO](ppo.md)** (classical RLHF) | Learned preference RM | Value-network baseline | PPO clip | Yes | General RLHF; still used at frontier labs |
| **[GRPO](grpo.md)** | Any scalar | Group-relative (K rollouts, z-score) | PPO clip | **No** | Modern default for reasoning RL / verifiable rewards |
| **[Online policy mirror descent](reasoning/online-policy-mirror-descent.md)** | Any scalar | Group-relative (K rollouts, mean only) | ℓ₂ on log-ratios (no clip) | No | Kimi k1.5's long-CoT RL; principled-derivation alternative to GRPO |
| **[RLVR](rlvr.md)** | Rule-based verifier | GRPO- or mirror-descent-style | Either | No | Math, code, format — anywhere correctness is checkable |
| **[Long-CoT RL](reasoning/long-cot-rl.md)** | Rule-based verifier | GRPO or mirror descent | PPO clip or ℓ₂ | No | Eliciting reasoning from a strong base without CoT SFT |
| **[DPO](dpo.md)** | Pairwise preferences | Closed-form | KL via log-ratio | No | Preference optimization without rollouts |
| **KTO / IPO / SimPO** | Unary or noisy preferences | DPO-variant (no depth file yet) | KL via log-ratio | No | Non-paired preference data or label noise |

All of these are the same policy-gradient skeleton with different specializations.

---

## How to choose

- **You have a verifier (math, code, format)** → [GRPO + RLVR](rlvr.md), or the mirror-descent alternative [online-policy-mirror-descent](reasoning/online-policy-mirror-descent.md). Cheapest, most stable, hardest-to-hack. Works from a strong SFT checkpoint or (for [long-CoT RL](reasoning/long-cot-rl.md)) from the base directly.
- **You have pairwise preferences, no interactive RM** → [DPO](dpo.md). No rollouts needed, one big offline pass.
- **You need open-ended helpfulness/tone shaping at scale** → Preference RM + PPO (classical RLHF). Still used by the frontier labs; rare in new open pipelines because DPO is simpler.
- **You have step-level labels for reasoning** → Consider a [PRM](reasoning/prm.md) used as a reranker (best-of-N). Using a PRM as a dense RL reward is brittle (see [prm](reasoning/prm.md) for why R1 abandoned it).
- **Small-model reasoning** → Skip RL, do **distillation SFT** from a larger reasoner's traces. R1's own experiments show this beats running RL directly on small models.

---

## Adjacent but distinct

- **Supervised fine-tuning** teaches *imitation* of reference completions. RL teaches *optimization* of a reward. Mixing the two (SFT then RL) is the dominant recipe.
- **Constitutional AI / RLAIF** replace the human preference labeler with an LLM judge. Still PPO/DPO under the hood; the novelty is in label generation, not the optimization algorithm.
- **Search at inference time** (MCTS, beam, best-of-N) is orthogonal — you can combine it with any of the RL techniques above. See [mcts](reasoning/mcts.md) and [orm](reasoning/orm.md) for the search+reward-model combination.
- **Imitation from demonstrations** (behavior cloning) is SFT. *Inverse* RL (recover the reward from demonstrations) is not common for LLMs — everyone just uses preferences.

---

## Sources

- Paper: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 — the PPO objective and clip.
- Paper: *Training language models to follow instructions with human feedback (InstructGPT)* — Ouyang et al., 2022 — the canonical SFT → RM → PPO pipeline, where "RL on LLMs" became standard.
- Paper: *Direct Preference Optimization* — Rafailov et al., 2023 — closed-form offline alternative to PPO for preference data.
- Paper: *DeepSeekMath* — Shao et al., 2024 — introduces [GRPO](grpo.md).
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — large-scale verifiable-reward RL, including the [long-CoT RL](reasoning/long-cot-rl.md) / R1-Zero result.
- Paper: *Tülu 3* — AI2, 2024 — canonical open recipe stacking SFT + DPO + [RLVR](rlvr.md).
- Textbook reference: Sutton & Barto, *Reinforcement Learning: An Introduction* (2018) — for the classical RL fundamentals (MDPs, policy gradient, value functions).

---

## Conventions

- **Filename:** `_rl.md` (leading underscore — taxonomy).
- **Folder placement:** `post-training/` root, same level as [_post-training.md](_post-training.md). `_post-training.md` is the broader map; `_rl.md` zooms into the RL subspace.
- **Scope:** concepts that appear in two or more RL-related depth files. Technique-specific mechanics live in the depth file, not here.
