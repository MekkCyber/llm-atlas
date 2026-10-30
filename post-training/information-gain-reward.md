# Information-Gain Reward
*Depth — RL reward derived from posterior-entropy drop over a hidden target, used to train question-asking / information-acquisition policies.*

**TL;DR:** An **information-gain reward** scores each agent action by how much it *reduces uncertainty* about a hidden target: `reward(a) = H(θ | h_{<t}) − H(θ | h_{<t}, a, obs)`. The target `θ` is latent (the answer to be identified, the diagnosis, the function signature), and the policy is trained to pick actions — most often **questions** — whose expected observation collapses posterior entropy fastest. Amortizing BED (Berthon, Astorga, van der Schaar, Cambridge, 2026) uses this reward with reasoning-model RL on Twenty Questions and MediQ, replacing inference-time Bayesian experimental design with a learned amortized policy.

**Prereqs:** [rlvr.md](rlvr.md), [_rewards.md](_rewards.md)
**Related:** [grpo.md](grpo.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [../agents/computer-use-agents.md](../agents/computer-use-agents.md)

---

## What it is

Sequential **Bayesian Experimental Design (BED)** picks the next experiment / query / question to maximize **expected information gain** about a latent variable `θ`:

$$
a^* = \arg\max_a \; \mathbb{E}_{obs \sim p(obs | a, h_{<t})} \big[\, H(\theta | h_{<t}) - H(\theta | h_{<t}, a, obs) \,\big]
$$

Classically done at inference time with Monte Carlo over the posterior — expensive, requires an explicit posterior model, and doesn't scale.

**Amortized BED** replaces this inference-time computation with a *learned policy* `π_θ(a | h_{<t})` trained to maximize information gain. The reward signal that trains it is the realized per-step entropy drop:

$$
r_t = H(\theta | h_{<t}) - H(\theta | h_{<t}, a_t, obs_t)
$$

Trained with RL (GRPO or similar). At inference, the policy acts in one forward pass per step — no posterior MC.

---

## How it works

### Training environment

```
sample θ ~ prior(Θ)                      # hidden target
loop:
    a_t = π(h_{<t})                      # policy acts (question)
    obs_t = env(θ, a_t)                  # oracle answer
    r_t = entropy_drop(θ, h_{<t}, a_t, obs_t)
    h_{<t+1} = h_{<t} ∪ (a_t, obs_t)
return trajectory (actions, rewards)
```

The *oracle* is a function of the hidden target (e.g., in Twenty Questions, "Is it a mammal?" → yes/no based on `θ`). Entropy is computed over the posterior `p(θ | h)`.

### Entropy computation

Two common forms:

- **Explicit posterior.** For small discrete `Θ`, maintain `p(θ | h)` exactly and compute `H` directly. Twenty Questions uses this.
- **Model-estimated posterior.** For continuous or large `Θ`, train a separate posterior-approximator (an auxiliary model) and use its outputs. MediQ (clinical dialogue) sits in this regime.

Either way, the entropy drop is a *verifiable* scalar — a specific instance of [RLVR](rlvr.md).

### Policy architecture

A reasoning LLM. The policy takes the dialogue history as input and emits a candidate action (natural-language question). GRPO rollouts, group-relative advantages, standard long-CoT RL — see [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md).

### Amortization payoff

Inference-time BED cost: `O(|θ| * K)` per step for a Monte Carlo posterior estimate with `K` samples.
Amortized policy cost: one LLM forward per step.

For reasonably sized `Θ`, amortization is 10×–1000× cheaper per step. Berthon et al. show the amortized policy also *outperforms* inference-time BED on Twenty Questions, suggesting the policy is finding non-trivial information-gain-maximizing strategies the brute-force MC misses.

---

## Why it matters

- **A principled, verifiable reward for information-acquisition tasks.** Diagnosis, debugging, interactive data analysis — tasks whose *point* is to extract information, not to produce a final answer immediately — now have a reward signal that matches the objective.
- **Fits straight into reasoning-model RL.** No new algorithm; GRPO + entropy-drop reward. The reward is a scalar, verifiable, cheap to compute for small `Θ`.
- **Transfers across tasks.** Berthon et al. report that policies trained on Twenty Questions transfer to unseen target categories and to clinical dialogue (MediQ) — the amortization generalizes.
- **A reward family missing from the standard taxonomy.** Prior RLVR rewards are mostly outcome-match (math answer, code tests, format). Information-gain is **step-wise** and **task-structural** — a distinct axis.

---

## Gotchas & tricks

- **Entropy must be computable or well-approximated.** For tasks with huge or ill-defined `Θ`, posterior approximation is expensive and biased, which biases the reward.
- **Reward hacking via oracle probing.** If the oracle is a trained model, the policy can learn adversarial questions that drive the oracle to high-entropy answers. Keep the oracle fixed and audited.
- **The policy can learn long "fishing" chains.** Without a cost per question, the policy may ask dozens before committing. Add a per-step cost or a budget.
- **Pairing with a terminal reward helps.** Pure entropy-drop reward optimizes for information acquisition, not final-answer correctness. Combine with an outcome reward to lock down the end of trajectory.
- **Need a strong base.** Weak bases can't produce informative questions. The amortized policy *refines* an existing question-asking ability; it does not create one from a model that has none.

---

## Sources

- Paper: *Amortizing Sequential Experimental Design in Reasoning LLMs* — Berthon, Astorga, van der Schaar, University of Cambridge, 2026, [arXiv 2610.20832](https://arxiv.org/abs/2610.20832).
- Classical BED: Chaloner & Verdinelli, *Bayesian Experimental Design: A Review*, 1995 — the inference-time precursor.
- Related: [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md) — the RL training backbone.
- Related: [_rewards.md](_rewards.md) — fits in alongside outcome, format, and process rewards.
