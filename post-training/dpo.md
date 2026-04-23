# Direct Preference Optimization (DPO)
*Depth — the closed-form, rollout-free alternative to PPO-RLHF for pairwise preference data.*

**TL;DR:** Start from the KL-regularized RLHF objective. Under Bradley-Terry pairwise preferences, there's a **closed-form relationship** between the optimal policy and the latent reward: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)`. Rearrange to write the reward as a log-ratio of policies, plug into the BT preference log-likelihood, and the reward model vanishes — you can train directly on preference pairs via a simple binary cross-entropy. **No reward model, no rollouts, no PPO, no value network.** Just one offline pass over `(prompt, preferred, rejected)` triples. Used by Zephyr, Tülu 3, Llama 3 alignment, and most modern open-recipe preference tuning.

**Prereqs:** [_post-training](_post-training.md), [_rl](_rl.md), [ppo](ppo.md)
**Related:** [grpo](grpo.md) · [rlvr](rlvr.md) · [_rewards](_rewards.md)

---

## What it is

Classical RLHF has three steps:
1. **SFT** on demonstrations.
2. **Reward model** `r_ϕ(x, y)` trained on pairwise preferences via Bradley-Terry MLE.
3. **RL** — PPO against `r_ϕ` with a KL penalty to the SFT checkpoint.

DPO collapses steps 2 and 3 into a single supervised loss. The claim that makes this work is: **for Bradley-Terry preference data, the policy you'd get from step 3 has a closed form in terms of the reward you'd have gotten from step 2, and you can reparameterize to eliminate the reward entirely.**

So DPO trains the *same* policy you'd get from PPO-RLHF, but in one offline pass without ever training a reward model or running rollouts. It's a preference-optimization technique, not a general RL algorithm — it only works when your signal is pairwise (or fully ranked) preference data.

---

## How it works

### Setup — the RLHF objective

Given:
- `π_θ` = trainable policy, `π_ref` = reference (usually the SFT checkpoint, frozen).
- Preference data `D = {(x_i, y_w, y_l)}` — preferred completion `y_w`, dispreferred `y_l`.
- Bradley-Terry model: preferences come from a latent reward `r*`:
  ```
  p*(y_1 ≻ y_2 | x) = σ( r*(x, y_1) - r*(x, y_2) )
  ```

Classical RLHF fits `r_ϕ` to the preferences (BT MLE), then optimizes:

```
max_{π_θ}  E_{x~D, y~π_θ} [ r_ϕ(x, y) ]  -  β · KL( π_θ( · | x) || π_ref( · | x) )
```

`β > 0` is the KL regularization strength.

### Step 1 — closed-form optimal policy

The maximizer of the KL-regularized RL objective has a Gibbs-Boltzmann form (derivation: rewrite the objective as a KL minimization; apply Gibbs' inequality):

```
π_r(y | x) = (1 / Z(x)) · π_ref(y | x) · exp( r(x, y) / β )

Z(x) = Σ_y  π_ref(y | x) · exp( r(x, y) / β )        (partition function)
```

This holds for *any* reward `r` — the Gibbs form is the KL-regularized optimal policy.

### Step 2 — solve for reward in terms of policy

Take logs and rearrange:

```
r(x, y) = β · log( π_r(y | x) / π_ref(y | x) )  +  β · log Z(x)
```

Every `(policy, π_ref)` pair **induces a reward**, up to a prompt-only shift `β · log Z(x)` (which doesn't depend on `y`).

### Step 3 — plug into the BT preference loss

The BT likelihood only depends on reward *differences*:

```
p*(y_w ≻ y_l | x) = σ( r*(x, y_w) - r*(x, y_l) )
```

The `β · log Z(x)` shift cancels (same term for `y_w` and `y_l`). Substituting the reward-as-log-ratio expression:

```
p*(y_w ≻ y_l | x) = σ( β · log(π*(y_w|x)/π_ref(y_w|x)) - β · log(π*(y_l|x)/π_ref(y_l|x)) )
```

### Step 4 — the DPO loss

Identify `π*` with the policy we're training, `π_θ`. Take the negative log-likelihood over the preference dataset:

```
L_DPO(π_θ ; π_ref) = - E_{(x, y_w, y_l) ~ D} [
    log σ( β · log( π_θ(y_w|x) / π_ref(y_w|x) )
         - β · log( π_θ(y_l|x) / π_ref(y_l|x) ) )
]                                                    (Rafailov 2023, Eq. 7)
```

This is a **binary cross-entropy** loss — same shape as a classifier. The reward model is gone: every term is a log-probability under `π_θ` or `π_ref`, both of which you can evaluate in one forward pass.

### The gradient

Let `r̂_θ(x, y) = β · log( π_θ(y|x) / π_ref(y|x) )` be the **implicit reward**. Then:

```
∇_θ L_DPO = -β · E_{(x, y_w, y_l) ~ D} [
    σ( r̂_θ(x, y_l) - r̂_θ(x, y_w) ) · ( ∇_θ log π_θ(y_w | x) - ∇_θ log π_θ(y_l | x) )
]
```

Three pieces:

- **`∇_θ log π_θ(y_w | x)`** — pushes up the likelihood of the preferred completion.
- **`-∇_θ log π_θ(y_l | x)`** — pushes down the likelihood of the rejected completion.
- **`σ(r̂_θ(x, y_l) - r̂_θ(x, y_w))`** — a per-sample weight that is *large* exactly when the implicit reward model **gets the pair wrong** (rates `y_l` above `y_w`). On correctly-ordered pairs, the sigmoid is small and the update shrinks toward zero.

The weighting is what makes DPO different from naive unlikelihood training (maximize log π(y_w) and minimize log π(y_l) unweighted) — the paper reports that naive unlikelihood causes model degeneration.

### Ranked generalization (Plackett-Luce)

For datasets with full rankings `τ` over `K > 2` candidates, DPO generalizes to a Plackett-Luce log-likelihood (Appendix A.3):

```
L_DPO = - E_{τ, y_1..K, x ~ D} [
    log ∏_{k=1..K}  exp(β · log(π_θ(y_{τ(k)}|x) / π_ref(y_{τ(k)}|x)))
                  / Σ_{j=k..K} exp(β · log(π_θ(y_{τ(j)}|x) / π_ref(y_{τ(j)}|x)))
]
```

`K = 2` recovers the pairwise loss. Rarely used in practice — pairwise is the default.

### Reference implementation

From Appendix B:

```python
pi_logratios  = pi_yw_logps  - pi_yl_logps
ref_logratios = ref_yw_logps - ref_yl_logps
losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
rewards = beta * (pi_logps - ref_logps).detach()   # implicit reward, for eval
```

That's it. Four lines of loss.

### Hyperparameters from the paper

- `β = 0.1` is the default (Anthropic HH, sentiment).
- `β = 0.5` for TL;DR summarization.
- Sentiment sweep used `β ∈ {0.05, 0.1, 1, 5}` to trace the reward-vs-KL frontier.
- Batch size 64, RMSprop, LR `1e-6` with linear warmup over 150 steps.
- `β` **large** → stays close to `π_ref` (strong KL). `β` **small** → free to move far (weak KL). In the DPO loss, `β` also sets the *margin* the sigmoid operates over — larger `β` ⇒ sharper per-example weighting.

---

## Why it matters

- **No reward model to train, serve, or tune.** A trained RM is a production surface (drifting distribution, recalibration, hackability). DPO deletes it. Every reward-over-optimization failure mode in classical RLHF traces back to the RM; DPO sidesteps that class entirely.
- **No rollouts during training.** The loss is evaluated on a fixed offline dataset. No inference-in-the-training-loop, no sampling-with-temperature, no KL-estimator variance, no PPO clip, no value network. The implementation fits in ~50 lines.
- **Stable and cheap.** Trains like SFT (one pass over preference pairs, standard optimizer). On the benchmarks in the paper (sentiment, summarization, HH dialogue), DPO matched or beat PPO-RLHF up to 6B parameters, with strictly better reward-vs-KL frontier in controlled sentiment comparisons.
- **The paper's theoretical bonus.** Section 5.2 shows that PPO's reward is missing the normalizer `β · log Σ_y π_ref(y|x) exp(r(x,y)/β)` — the *soft value function*. Classical RLHF works around this with human-completion baselines (Monte-Carlo estimates). DPO's reparameterization absorbs the normalizer automatically, so no baseline is needed.
- **The default open-recipe preference tuning.** Zephyr, Tülu 3, OpenHermes, most HuggingFace alignment pipelines, and Llama 3's alignment use DPO (or a DPO variant) rather than PPO. The ergonomics won.
- **Composes with RL.** You can chain: SFT → DPO → RLVR (see [Tülu 3]) — DPO shapes style and preferences cheaply, then RL pushes capability on verifiable tasks.

---

## Gotchas & tricks

- **Length bias is real.** DPO tends to reward longer preferred completions, because the log-ratio scales with sequence length. Preference datasets where `y_w` is systematically longer than `y_l` will push the model to verbose outputs. Mitigations: length-controlled DPO, length-normalized log-probabilities, or preference-pair length matching at collection time.
- **`π_ref` must match the preference distribution.** If `π_ref ≠ π_SFT`, the implicit reward estimate is biased. The paper recommends `π_ref = π_SFT`. When no SFT is available (HH), they suggest fitting `π_ref` by SFT on the *preferred* completions only — crude but works.
- **Out-of-distribution coverage depends on the preference set.** DPO has no way to generate new rollouts during training, so whatever distribution you preference-labeled is the distribution you can tune. PPO's online rollouts let the policy explore regions the preference data didn't cover; DPO can't. For narrow preference data this shows up as poor generalization.
- **β is a global knob, not a schedule.** Unlike PPO's adaptive KL, DPO's β is fixed. Some follow-ups (β-scheduled DPO) anneal it; the original doesn't.
- **Naive unweighted unlikelihood loss degenerates.** Dropping the `σ(r̂_θ(x, y_l) - r̂_θ(x, y_w))` weight and just maximizing `log π(y_w) - log π(y_l)` unweighted causes mode collapse and nonsensical output. The dynamic weighting is load-bearing.
- **Implicit reward `r̂_θ` is interpretable.** `r̂_θ(x, y) = β · log(π_θ(y|x) / π_ref(y|x))` can be logged during training to monitor convergence — rising `r̂(y_w) - r̂(y_l)` margin means the policy is learning.
- **No guarantee `y_w` gets absolutely more probable.** DPO can push `log π(y_w)` *down* as long as `log π(y_l)` goes down faster. Watch the absolute log-prob of preferred completions, not just the margin. Some deployments see both go down — the model is optimizing the ranking, not the likelihood.
- **Can be brittle to preference noise.** High label noise in preferences makes the BT assumption fragile. Variants: IPO (Identity Preference Optimization, regression form, less extreme), cDPO (conservative DPO with a noise term), KTO (doesn't assume paired data).
- **Reward over-optimization exists differently in DPO.** Classical RLHF over-optimization happens against a trained RM. DPO's equivalent is over-optimizing against the implicit reward implied by `π_ref` — if the preference set is inconsistent with `π_ref`'s distribution, the policy can over-fit the preference data at the cost of general capability. The paper's Fig. 3-right shows a slight drop at high training; follow-ups study this as a DPO-specific phenomenon.
- **SFT → DPO works better than DPO alone.** The paper uses SFT as a prerequisite in most experiments. Skipping SFT and going straight to DPO on a base model rarely works well.
- **Scale beyond 6B is less studied in the original paper.** DPO's original evaluation maxed at 6B parameters. Follow-ups (Tülu 3, Zephyr-141B) have pushed it much further successfully, but the original paper did not demonstrate frontier-scale.

---

## Follow-up variants (flagged, not depth files yet)

- **IPO** (Azar et al. 2023) — identity preference regression, avoids DPO's tendency to push log-ratios to infinity on clean preferences.
- **KTO** (Kahneman-Tversky Optimization, Ethayarajh et al. 2024) — works with unary feedback (thumbs up/down) instead of paired preferences. **no depth file yet**
- **cDPO / rDPO** — conservative/robust variants that add a noise model to the BT assumption. **no depth file yet**
- **SimPO** (Meng et al. 2024) — length-normalized DPO without a reference model. **no depth file yet**
- **ORPO** (Hong et al. 2024) — combines SFT and preference optimization in one step via an odds-ratio loss. **no depth file yet**

If any of these become primary reads, they should graduate to their own depth file per the "every concept needs a depth file" rule.

---

## Sources

- Paper: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — Rafailov, Sharma, Mitchell, Ermon, Manning, Finn, NeurIPS 2023, arXiv 2305.18290 — the original DPO paper.
- Paper: *Training language models to follow instructions with human feedback (InstructGPT)* — Ouyang et al., 2022, arXiv 2203.02155 — the classical RLHF pipeline DPO shortcuts.
- Paper: *Zephyr: Direct Distillation of LM Alignment* — Tunstall et al., 2023, arXiv 2310.16944 — early demonstration of DPO on 7B instruction tuning.
- Paper: *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* — AI2, 2024 — canonical open recipe stacking SFT + DPO + RLVR.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783 — Llama 3 alignment uses DPO-family losses.
- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI, 2025 — one of its long2short variants (see [long2short](reasoning/long2short.md)) is a specific DPO formulation on shortest-correct positives vs long-correct and incorrect negatives.
