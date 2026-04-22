# Sequence-Wise Load-Balance Loss
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The same functional form as the [vanilla load-balancing loss](load-balancing-loss.md), but with statistics `f_i` and `P_i` computed **over a single sequence** (`T` = sequence length) instead of the whole batch. This enforces balance *within* each sequence rather than averaged *across* sequences — a strictly stronger constraint. First introduced under a different name ("expert-level balance loss") in DeepSeekMoE (Dai 2024) and explicitly called *sequence-wise* in DeepSeek-V3, where it's used at a very small coefficient (`α = 10⁻⁴`) as a safety net on top of aux-loss-free balancing.

**Prereqs:** [_moe](_moe.md), [load-balancing-loss](load-balancing-loss.md)
**Related:** [aux-loss-free-balancing](aux-loss-free-balancing.md) · [deepseek-moe](deepseek-moe.md)

---

## What it is

Vanilla load-balancing loss (GShard / Switch) computes `f_i = fraction of tokens dispatched to expert i` over the entire batch of tokens. Under this "batch-wise" scope, a single sequence can route all its tokens to expert 7 as long as some other sequence in the same batch uses the other experts — average balance across the batch remains fine.

For fine-grained MoE (hundreds of small experts) this batch-wise slack is a real problem: one sequence concentrating on a handful of experts hurts expert parallelism (those experts become bandwidth hotspots), reduces per-sequence specialization diversity, and can let the router silently develop sequence-specific routing patterns that hurt generalization.

Sequence-wise balance loss tightens the scope: the sum `Σ_t` in `f_i` and `P_i` is taken over tokens of *one sequence* only. A separate balance penalty is computed for every sequence. Balance is enforced everywhere, not just on average.

---

## How it works

### The DeepSeek-V3 form (explicit statement, Eqs. 17–20)

For a single sequence with `T` tokens, `N_r` routed experts, top-`K_r` routing, and affinity scores `s_{i,t}` (sigmoid-based — see [DeepSeekMoE](deepseek-moe.md)):

```
f_i = (N_r / (K_r · T)) · Σ_{t=1..T}  1{ s_{i,t} ∈ Top-K_r({s_{j,t}}_{j=1..N_r}) }

s'_{i,t} = s_{i,t} / Σ_{j=1..N_r}  s_{j,t}

P_i = (1/T) · Σ_{t=1..T}  s'_{i,t}

loss_seq  =  α · Σ_{i=1..N_r}  f_i · P_i
```

The normalization constants make `f_i = 1/N_r` and `P_i = 1/N_r` under uniform routing, so `Σ_i f_i · P_i = 1/N_r` and the loss is `α/N_r` at perfect balance.

The difference from Switch's Eq. 4–6 is subtle but critical: **`T` is the number of tokens in a single sequence**, not the batch. DeepSeek-V3 is explicit about this right after Eq. 20: *"T denotes the number of tokens in a sequence."*

### Per-sequence, per-layer

The loss is summed over sequences in the batch and over MoE layers:

```
L_seq_total  =  Σ_{ℓ ∈ MoE layers}  Σ_{s ∈ sequences in batch}  loss_seq^{(ℓ, s)}
```

Every (layer, sequence) pair gets its own balance penalty. The per-sequence statistics are small — `T` might be 4K or 32K tokens instead of a batch of millions — so variance is higher and the signal is noisier. The coefficient must be small enough to absorb this without interfering with the task loss.

### Contrast with vanilla (batch-wise)

| | Vanilla (GShard / Switch) | Sequence-wise |
|---|---|---|
| Scope of `f_i` / `P_i` sums | over all tokens in the batch | over tokens of one sequence |
| Number of balance terms per layer | 1 per batch | 1 per sequence × per layer |
| Constraint strength | balance on batch average | balance within every sequence |
| Statistic variance | low (many tokens) | high (few tokens) |
| Workable coefficient `α` | ~10⁻² (Switch) | ~10⁻³–10⁻⁴ (DeepSeekMoE / V3) |

A batch with one all-code sequence and one all-prose sequence: vanilla aux loss is satisfied if code-tokens go to experts 1–8 and prose-tokens go to experts 9–16 (each expert gets `T/16` tokens across the batch). Sequence-wise is *not* satisfied — it requires within-code-sequence balance *and* within-prose-sequence balance, forcing code tokens to spread across all experts (losing some specialization).

### Why DeepSeek-V3 uses it at `α = 10⁻⁴`

The paper is explicit about the tiny coefficient. Quote (§4.2): *"we set α to 0.0001, just to avoid extreme imbalance within any single sequence."* The aux-loss-free bias mechanism does the real load-balancing work at batch scope; the sequence-wise term is a guardrail against catastrophic single-sequence concentration, not a primary balancer. At this scale the gradient interference is negligible.

Contrast DeepSeekMoE (where this loss is the primary balancer, no aux-loss-free):
- `α_1 = 0.01` for 2B-scale validation
- `α_1 = 0.001` for DeepSeekMoE 16B
- `α_1 = 0.003` for DeepSeekMoE 145B

Orders of magnitude larger when it's the main mechanism.

---

## Why it matters

- **Closes the single-sequence loophole.** Vanilla aux loss allows per-sequence concentration; sequence-wise doesn't. For fine-grained MoE with hundreds of experts, this gap is non-trivial.
- **Cheap safety net on top of aux-loss-free.** At `α = 10⁻⁴` the gradient interference is below noise; you get the single-sequence protection essentially for free.
- **The actual primary balancer in DeepSeekMoE.** Before aux-loss-free was invented, this (under the name "expert-level balance loss") was what kept DeepSeekMoE's 256-expert routing from collapsing.

---

## Gotchas & tricks

- **Stricter balance ⇒ less specialization.** A sequence of pure code tokens, under sequence-wise balance, must spread its routing across all experts — even though sending code tokens to non-code experts is semantically wrong. Large `α` in a sequence-wise formulation can hurt domain specialization.
- **High variance of the statistic.** A 4K-token sequence computing `f_i` over 256 experts has on average 4000·8/256 ≈ 125 tokens per expert. This is a noisy estimate. Very large `α` amplifies the noise into the router gradient.
- **Don't reuse a batch-wise `α`.** Switch's `α = 10⁻²` is for a batch statistic with `T` in the millions. Applied to sequence-scope, the same coefficient is roughly 2–3 orders of magnitude too strong for the signal quality.
- **The `N_r / (K_r · T)` normalization is non-obvious.** It's chosen so that under perfectly uniform top-`K_r` routing, `f_i = 1/N_r` exactly — match-up with `P_i`'s `1/N_r` under uniform sigmoid scores, so the product-and-sum lands on `1/N_r` at optimum.
- **Don't combine with vanilla aux loss without thought.** Stacking a batch-wise aux loss and a sequence-wise aux loss double-penalizes balance with mostly the same signal. Pick one as primary or keep both at very small coefficients.
- **Sigmoid vs softmax matters.** DeepSeek's sigmoid-per-expert router gives `s_{i,t}` that don't sum to 1 across experts; the `s'_{i,t} = s_{i,t} / Σ_j s_{j,t}` normalization step (Eq. 19) is what turns them into a probability-like quantity for `P_i`. A softmax router skips this step (it's already normalized). If you copy the formula onto a softmax router, drop Eq. 19.

---

## Sources

- Paper: *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* — Dai et al., 2024 — Sec 3.3, Eqs. 12–14. First appearance of the formula, under the name "Expert-Level Balance Loss". `T` is explicitly sequence length (Sec 3.1).
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — Sec 2.1.2, Eqs. 17–20 and Sec 4.2. Introduces the label "sequence-wise", adds the sigmoid-normalization step, reduces `α` to `10⁻⁴` as a safety net on top of aux-loss-free balancing.
- Paper: *Switch Transformers* — Fedus et al., 2021 — the batch-scope form this variant tightens. Sec 2.2, Eqs. 4–6.
