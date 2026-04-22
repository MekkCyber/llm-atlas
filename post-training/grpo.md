# Group Relative Policy Optimization (GRPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A simplified PPO variant that removes the value (critic) network. For each prompt, sample `G` responses from the current policy, compute each response's reward, then use the **group's mean and standard deviation as a baseline** to normalize advantages. The rest of the update is standard PPO — clipped ratio with a KL penalty toward a reference model. Introduced in DeepSeekMath (2024), now the default policy-optimization algorithm for reasoning RL and RLVR pipelines.

**Prereqs:** basic RL (policy gradients, PPO), KL divergence — see [_rl](_rl.md) for a walk-through.
**Related:** [rlvr](rlvr.md) · [long-cot-rl](reasoning/long-cot-rl.md) · [_rewards](_rewards.md) · [deepseek-r1 case study](../case-studies/deepseek-r1.md)

---

## What it is

Standard **PPO** for LLMs pairs a policy network (the model being trained) with a **value network** — a separate copy of the model with a scalar-output head that predicts the expected reward-to-go at each token position. The value network is used as the advantage baseline:

```
A_t = R_t - V(s_t)              ← advantage, reduces variance of policy gradient
```

The value network is expensive: you train a second model of comparable size, with its own loss, and you pay its forward/backward cost on every RL step. For LLMs where the policy already has 10s to 100s of billions of parameters, the value network ~doubles RL compute.

**GRPO's insight:** if you always sample **multiple responses per prompt**, you can replace the value-network baseline with the **empirical group mean**. Cheaper, simpler, and often a better baseline for the sparse-reward / verifiable-reward setting that reasoning RL uses.

---

## How it works

### The sampling step

For each prompt `q` in the RL batch, sample `G` full responses from the current policy `π_θ^old`:

```
{ o_1, o_2, ..., o_G } ~ π_θ^old( · | q )
```

`G = 4` to `G = 64` typical; `G = 16` is common. Higher `G` gives lower-variance advantage estimates but multiplies per-prompt rollout cost.

### Compute rewards

Each response gets a scalar reward `r_i`:

```
r_i = R(q, o_i)
```

Where `R` is whatever reward function you use — rule-based verifier (for RLVR: math answer match, unit test pass, format check), model-based reward (for RLHF-style), or some combination. Rewards are per-response, not per-token.

### Group-relative advantage

Compute the group's reward statistics:

```
r̄ = (1/G) · Σ_i r_i              ← mean
σ_r = std(r_1, ..., r_G)

A_i = (r_i - r̄) / σ_r             ← normalized advantage, broadcast to all tokens in o_i
```

This `A_i` is the advantage assigned to **every token** of response `o_i`. It's a single scalar per response, not per-token — all tokens in a response share the same advantage signal.

The normalization by `σ_r` matters: it makes the RL update invariant to the reward function's absolute scale. If all responses get very similar rewards (σ_r ≈ 0), the advantages are small (noisy) and the policy barely moves — exactly the right behavior.

### PPO-clipped policy update

With advantages in hand, the policy update is standard PPO:

```
ratio_{i,t}(θ) = π_θ(o_{i,t} | q, o_{i,<t}) / π_θ^old(o_{i,t} | q, o_{i,<t})

L_CLIP = - (1/G) · Σ_i (1/|o_i|) · Σ_t  min(
    ratio_{i,t} · A_i,
    clip(ratio_{i,t}, 1-ε, 1+ε) · A_i
)
```

where `ε = 0.2` is the PPO clip parameter, standard.

### KL regularization to a reference model

Add a KL penalty between the current policy and a fixed **reference model** `π_ref` (usually the SFT checkpoint or the pretraining base):

```
L_KL = β · (1/G) · Σ_i (1/|o_i|) · Σ_t  KL( π_ref( · | ...)  ||  π_θ( · | ...) )
```

`β = 0.01` to `0.1` typical. The KL term prevents the policy from drifting too far from `π_ref` and losing general capabilities.

### Full objective

```
L_GRPO = L_CLIP + L_KL
```

There's no value-function loss (no critic to train) and no explicit entropy bonus (the KL to `π_ref` handles exploration implicitly).

### Token-level vs response-level rewards

GRPO as written assigns the same `A_i` to every token of response `o_i`. This is fine for verifiable-reward setups where only the final answer matters (math: is the answer correct; code: do the tests pass). For settings where some tokens contribute more than others (e.g. preference-reward models that can score partial outputs), you'd want a token-level advantage — GRPO isn't designed for that; use PPO with a value network instead.

---

## Why it matters

- **Half the RL compute of PPO** for LLMs. Removing the value network saves its forward/backward pass on every step — a massive fraction of RL step cost when the policy has tens of billions of parameters.
- **Better baseline for sparse rewards.** The group mean is empirically a better variance-reduction baseline than a learned value network when rewards are sparse (binary correct/incorrect) and the value network has little to learn from. Especially true for reasoning RL.
- **No value-head warmstart problem.** PPO's value network needs its own warmup and can be unstable early in RL training (its predictions are bad before it's seen many rollouts). GRPO sidesteps this entirely.
- **Standard RL algorithm for reasoning.** Used in DeepSeekMath, DeepSeek-R1, DeepSeek-V3's post-training, and nearly every subsequent reasoning-RL paper in 2024–2025. When people say "RL with verifiable rewards," they mean GRPO (or a close variant) with rule-based rewards.
- **Runs the entire reasoning pipeline of R1-class models.** DeepSeek-R1 uses GRPO in **both** RL stages — Stage 2 (reasoning-oriented RL on a cold-start SFT checkpoint, with rule-based accuracy + format + **language-consistency** rewards summed directly) and Stage 4 (all-scenarios RL combining rule-based rewards for reasoning with learned preference RMs for helpfulness/harmlessness). R1-Zero further shows GRPO + rule rewards can drive reasoning *from a pretrained base*, no SFT at all — see [long-cot-rl](reasoning/long-cot-rl.md).

---

## Gotchas & tricks

- **`G` is a tradeoff.** Small `G` (4, 8): cheap but noisy advantages. Large `G` (64, 128): low-variance but expensive rollouts. `G = 16` is the common default.
- **Normalization by `σ_r` can blow up.** If all `r_i` are identical, `σ_r = 0` and dividing is undefined. Standard implementations add a small epsilon or fall back to `A_i = r_i - r̄` (no normalization).
- **KL term is non-negotiable.** Without it, the policy collapses to reward-hacking outputs that maximize the verifier but are garbage in general capability. `β` in the 0.01–0.1 range; tune downward over training as the policy stabilizes.
- **Reward scaling matters.** Because `σ_r` normalizes within a group, the absolute reward scale doesn't matter *within a prompt*, but if different prompts have wildly different reward magnitudes, the contribution to the gradient is uneven. Sometimes people add an outer normalization across the batch, but the paper doesn't.
- **Advantages are per-response, broadcast over tokens.** Not per-token. This is simpler to implement than PPO's per-token advantages but gives up the ability to distinguish which tokens in a response were the important ones. For verifiable rewards this doesn't matter; for fine-grained rewards it can.
- **Off-policy drift.** The `π_θ^old` you sample from is typically the policy at the start of the current step (or a few steps ago in mini-batched updates). After enough inner optimization steps, the ratio `π_θ / π_θ^old` drifts, and the PPO clip starts constraining every update. Small mini-batch counts (1–4 inner steps) are typical.
- **Entropy collapse.** Without an explicit entropy bonus, the policy can sharpen to near-deterministic — bad for exploration and sometimes bad for quality. If you see acceptance rates dropping or all responses converging, either add a small entropy bonus or increase the KL coefficient.
- **Works best with verifiable rewards.** GRPO's variance-reduction trick assumes the reward function is cheap to evaluate on many samples per prompt. For expensive reward models (a large learned RM queried per response), the `G`-way rollout becomes the bottleneck and PPO-with-value-net may be cheaper.
- **Don't confuse with REINFORCE with baseline.** REINFORCE is single-sample; GRPO is multi-sample with group baseline. The sampling structure is the whole point.
- **Composite rewards compose by direct sum.** Papers that combine signals (e.g., R1's *accuracy + format + language-consistency*) sum them before computing `A_i`. The group normalization then handles the overall scale. Note this is a real alignment-vs-capability lever: R1 observed that adding the language-consistency term slightly reduces benchmark scores but produces much cleaner CoT — the tradeoff is worth naming explicitly.
- **Hyperparameters from the original paper are unspecified at scale.** DeepSeek-R1 uses GRPO but doesn't disclose `G`, `ε`, or `β` for the R1 runs. For reproduction, open implementations (veRL, TRL, OpenRLHF) default to `G ∈ {8, 16}`, `ε = 0.2`, `β ∈ {0.001, 0.04}`. Treat these as starting points, not canonical values.

---

## Sources

- Paper: *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — Shao et al., DeepSeek, 2024 — introduces GRPO.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — applies GRPO with hybrid rule-based + model-based rewards.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — large-scale reasoning RL using GRPO with verifiable rewards; R1-Zero uses GRPO from base with no SFT, R1 uses GRPO in Stages 2 and 4 of a 4-stage pipeline.
- Paper: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 — the PPO baseline GRPO simplifies.
