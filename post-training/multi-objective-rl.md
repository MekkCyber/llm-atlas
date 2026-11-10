# Multi-Objective RL for LLMs
*Depth — training a steerable Pareto family of policies from a single RL run via per-objective adapters.*

**TL;DR:** Classical RLHF collapses many objectives (helpfulness, harmlessness, honesty, format, safety) into a single scalar reward, which hides the Pareto frontier and forces you to retrain for every mix. The adapter-per-objective recipe (LambdaMix is the clean example) trains one low-rank adapter per objective against its own normalized reward, then exposes a preference vector at serve time that both selects the adapter mix *and* reweights the rewards during training. One run, whole frontier.

**Prereqs:** [grpo](grpo.md), [dpo](dpo.md), [fine-tuning/lora.md](fine-tuning/lora.md), [_rewards](_rewards.md)
**Related:** [rlvr](rlvr.md), [ppo](ppo.md), [_rl](_rl.md)

---

## What it is

You have `K` reward signals `r_1, …, r_K` (e.g. helpfulness RM, harmlessness RM, format checker, length regularizer). You want a *family* of policies, each specialized to one convex combination `λ ∈ Δ^K` of those rewards. Three approaches exist:

1. **Scalarization.** Pick one `λ`, sum rewards, train one policy with PPO/GRPO. Simple, but gives one point on the frontier per run.
2. **Conditioned policy.** Train a single policy that reads `λ` as an input. Works, but the policy's conditioning on `λ` is implicit and often weak.
3. **Adapter-per-objective (modern default).** Freeze a base model; add one LoRA adapter per objective; train each adapter with GRPO against its own reward on the policy mixture determined by `λ`. At inference, the preference vector selects the adapter weighting *and* the training reward weighting.

Option (3) is the LambdaMix recipe. Because the adapters are composable (LoRA sums linearly), the inference-time policy is a smooth function of `λ`, and because each adapter trains against its own reward's normalization, cross-objective gradient interference is reduced.

## How it works

### Setup

- Frozen base `π_0`.
- `K` LoRA adapters `A_1, …, A_K` with per-adapter parameters `θ_k`.
- `K` reward functions `r_k`, each independently normalized (z-scored over its own training batch).
- Preference vector `λ ∈ Δ^K`, sampled per rollout during training.

### Rollout

With `λ` sampled from a prior over the simplex:

```
θ_mix(λ) = Σ_k λ_k · θ_k
π_λ = π_0 + LoRA(θ_mix(λ))
y ~ π_λ(·|x)
```

### Group-relative rewards, per objective

GRPO-style groups of `G` responses per prompt. For each adapter `k`, compute the group-mean and group-std of `r_k` *within the group*, giving a normalized per-objective advantage `A_{k,i}`.

### The multi-objective update

Each adapter `k` gets a gradient weighted by both the preference `λ_k` *and* its own per-objective advantage:

```
∇θ_k ← λ_k · (1/G) Σ_i A_{k,i} · ∇ log π_λ(y_i | x)
```

Over many rollouts with varied `λ`, each adapter concentrates on improving its own reward across the region of preference space it sees, while the shared base stays frozen.

### Inference-time steering

At serve time, pick `λ` and mix adapters:

```
π_serve(λ) = π_0 + LoRA(Σ_k λ_k · θ_k)
```

This is a cheap deterministic operation (adapter sum + merge). You can expose `λ` as a user-facing knob, pick it per request type, or tune it online.

## Why it matters

- **One run for the whole frontier.** Scalarized RLHF gives one Pareto point per training run. LambdaMix-style methods give a smooth family at roughly the cost of a single multi-task run.
- **Composable alignment.** You can train objectives independently and later mix them — the organization doing helpfulness tuning doesn't need to coordinate with the team doing harmlessness tuning.
- **Smaller training footprint.** LoRA adapters are tiny; you can afford one per objective and still fit the optimizer state on a single node.
- **Per-user or per-segment deployment.** Different users / products / jurisdictions can serve different `λ` from the same base. Regulated settings (clinical, legal) can crank up the relevant objective at inference.
- **Interpretability dividend.** The per-adapter gradient reveals which parameters each objective moves — an auxiliary interpretability signal for alignment research.

## Gotchas & tricks

- **Reward normalization has to be per-objective.** Sharing normalization across rewards defeats the "independent axes" claim and reintroduces reward-magnitude bias. Each `r_k` gets its own moving mean and std.
- **Preference sampling matters.** Uniform over the simplex trains well but under-visits corners (pure helpfulness, pure harmlessness). Dirichlet sampling or importance-weighting toward corners helps if you care about the extremes.
- **Adapter rank controls interference.** Low ranks force adapters to live in a small subspace and inherit interference; high ranks are more expressive but more expensive. Typical: `r = 8–32` per adapter.
- **Not a free lunch for conflicting objectives.** If `r_1` and `r_2` are antagonistic (helpfulness that reveals too much vs harmlessness that refuses), no amount of adapter tuning erases the Pareto tradeoff — it just lets you *see* it.
- **KL regularization is still needed.** Each adapter's update should be regularized toward the base model (or its SFT checkpoint) exactly as in single-objective GRPO; otherwise individual adapters mode-collapse.
- **At inference, be explicit about `λ`.** User-facing `λ` defaults should be documented; silent middleware changes to `λ` are a safety surface.

## Sources

- Paper: *Steerable Multi-Objective Group-Relative Preference Optimization with Jointly Trained Low-Rank Adapters* — Karouzos & Aletras, 2026 — arXiv 2611.09634 — the LambdaMix recipe this depth file is anchored on.
- Paper: *Rewarded Soups: Towards Pareto-Optimal Alignment by Interpolating Weights Fine-Tuned on Diverse Rewards* — Ramé et al., 2023 — adjacent weight-interpolation Pareto approach.
- Paper: *Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization* — Zhou et al., 2024 — MO-DPO, the preference-only analogue.
