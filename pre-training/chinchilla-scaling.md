# Chinchilla Scaling Laws
*Depth — the compute-optimal relationship between model size N and training tokens D.*

**TL;DR:** Hoffmann et al. (DeepMind, 2022) showed that under a fixed compute budget C, **model size N and training tokens D should scale equally**: `N ∝ C^0.5`, `D ∝ C^0.5`. At Gopher-scale compute, this gives roughly **D ≈ 20 · N** tokens per parameter. Proven by training **Chinchilla (70B, 1.4T tokens)** and showing it beats Gopher (280B, 300B tokens) at the same compute. Reversed the Kaplan 2020 conclusion that model size should grow faster than data. The reason every post-2022 frontier model is "more data, same or smaller parameter count."

**Prereqs:** [_parallelism](_parallelism.md) (for the compute budget framing)
**Related:** [downstream-scaling-laws](downstream-scaling-laws.md) · [mid-training](mid-training.md)

---

## What it is

A family of empirical laws characterizing how pretraining loss `L(N, D)` depends on model parameter count `N` and training token count `D`. The crucial derived result: for a given compute budget `C ≈ 6 · N · D`, the compute-optimal `(N*, D*)` satisfies:

```
N*(C) ∝ C^0.5
D*(C) ∝ C^0.5
```

Equivalently: **double compute → double both N and D**. Not "double N at fixed D" (Kaplan 2020's prescription) and not "double D at fixed N" (an even earlier intuition).

The paper derived this three independent ways (§3.1, §3.2, §3.3), all three converging on the same conclusion: prior models like GPT-3, Gopher, Megatron-Turing-NLG were **substantially under-trained** given their compute budget. Making them smaller and training them longer at the same compute would give a lower loss.

### The validation experiment

- **Gopher**: 280B parameters, trained on 300B tokens. Compute budget ≈ 5.76 × 10²³ FLOPs.
- **Chinchilla**: 70B parameters (4× smaller), 1.4T tokens (4.7× more), **same compute**.

Result: Chinchilla beats Gopher on essentially every benchmark — MMLU, BIG-bench, reading comprehension, math — with 4× fewer parameters. The paper reports **MMLU 67.5%** (vs Gopher's 60%), >7 pp absolute improvement (Abstract, §4.2).

This single experiment demolished the "bigger is always better at fixed compute" intuition that dominated 2020–2022.

---

## How it works

### Setup — the compute budget

The paper assumes dense Transformer training. FLOPs for one token:
- Forward: `2 · N` (matmul-dominated).
- Backward: `2 × forward = 4 · N`.
- Total: `6 · N` FLOPs per token.

So total training compute:

```
C ≈ 6 · N · D   FLOPs
```

This is the basis for every scaling-law plot in the paper: compute is treated as a single scalar, N and D trade off under the constraint C = const.

### Approach 1 — Fix model size, vary tokens (§3.1)

Train a family of models from 70M to 10B parameters. For each model, train with 4 different cosine LR schedule lengths (ratio 16× between shortest and longest). Every run produces a loss curve as a function of training FLOPs. For each compute budget, pick the model/LR-schedule that achieves lowest loss. Fit power laws.

Result: `N*(C) ∝ C^0.50`, `D*(C) ∝ C^0.50`. (Table 2, row 1.)

### Approach 2 — IsoFLOP profiles (§3.2)

Fix 9 compute budgets: `6 × 10^18, 1 × 10^19, …, 3 × 10^21` FLOPs. For each, train many models of different sizes, each reaching that FLOP count (so `D = C / (6N)`). Plot final loss vs N for each IsoFLOP curve — each curve has a minimum at some optimal N. Fit a parabola per curve, locate the minimum. Fit power laws across the 9 budgets.

Result: `N*(C) ∝ C^0.49`, `D*(C) ∝ C^0.51`. (Table 2, row 2.)

### Approach 3 — Parametric fit (§3.3)

Pool all `(N, D, L)` points from Approaches 1 and 2 and fit a three-term parametric law:

```
L(N, D) = E + A / N^α + B / D^β
```

- **E**: irreducible loss (entropy of natural text under the modeling distribution). Cannot be reduced by any (N, D).
- **A / N^α**: parameter-count term. Drops as N grows — the gap to an "ideal generative process" under the given parameter class.
- **B / D^β**: data term. Drops as D grows — the gap from finite training.

Fitted values (Equation 10, Appendix D.2):

```
E = 1.69
A = 406.4
B = 410.7
α = 0.34
β = 0.28
```

So: `L(N, D) ≈ 1.69 + 406.4 / N^0.34 + 410.7 / D^0.28`.

Compute-optimal point: minimize `L(N, D)` subject to `6·N·D = C`. Lagrange multiplier; result (§3.3, Equation 4):

```
N_opt(C) = G · (C/6)^a
D_opt(C) = G^{-1} · (C/6)^b

with
G = ((αA) / (βB))^{1/(α+β)}
a = β/(α+β) ≈ 0.46
b = α/(α+β) ≈ 0.54
```

Approach 3's exponents are slightly different from Approaches 1 and 2 but all three are within ±0.05 of 0.5 — **the conclusion "N and D should scale roughly equally with compute" is robust**.

### The empirical ratio D/N

At Gopher-scale compute (~6 × 10²³ FLOPs), the optimal ratio lands near:

```
D*/N* ≈ 20 tokens per parameter
```

For Chinchilla: 1.4T / 70B = **20 tokens/parameter**. For Gopher: 300B / 280B ≈ **1.1 tokens/parameter** — 20× under-trained.

Note that `D*/N*` is not strictly constant — it grows slowly with compute because α and β aren't exactly equal. But in the 10²²–10²⁴ FLOP range, **20× is a reasonable rule of thumb**.

---

## Historical context

### Kaplan et al. 2020 — the contrast

"Scaling Laws for Neural Language Models" (Kaplan et al., 2020, arXiv 2001.08361) analyzed the same question but concluded:

```
N*(C) ∝ C^0.73
D*(C) ∝ C^0.27
```

— model size should grow **much faster** than data. This motivated the 2020–2022 industry pattern: bigger and bigger models (GPT-3 175B, MT-NLG 530B, Gopher 280B) on similar-sized datasets (300B–500B tokens).

Chinchilla's Section 3.1 explains what Kaplan got wrong: their study used a **fixed learning rate schedule** across all model sizes. When you train a small model with a LR schedule calibrated for 10B parameters, its loss at the end of training is worse than it would be with a properly-tuned schedule. This biases the "small models underperform" signal, pushing the optimal-N estimate higher.

Chinchilla's Approaches 1 and 2 fix this by varying the LR schedule length with the model size. Approach 3 side-steps it entirely with a parametric fit. All three land at `N* ∝ C^0.5`.

### The post-Chinchilla world

Every 2023+ frontier model embraces Chinchilla-style ratios:

- **Llama 2**: 7B/13B on 2T tokens (**~300:1 and 150:1**). Llama models are deliberately trained **past compute-optimal** — for inference efficiency, smaller models are trained longer.
- **Llama 3 8B**: trained on 15T tokens (**1875:1**). Extreme over-training.
- **Llama 3 405B**: 15.6T tokens (**~38:1**). Near-compute-optimal (Llama 3 paper predicts 402B on 16.55T from their own scaling laws).
- **DeepSeek-V3**: 671B total (37B active MoE) on 14.8T tokens. At 37B active: **~400:1**.
- **Chinchilla**: 70B on 1.4T (**20:1**) — the reference point.

Compute-optimal is the *floor*; over-training trades compute-at-train for quality-at-inference, which is usually a good trade for deployed models.

---

## Why it matters

- **Calibrates resource planning.** Given compute C, picking `N, D` correctly matters more than optimizing optimizer tricks. Chinchilla's ratio is the starting point.
- **Shifted the field from "more parameters" to "more tokens."** The 2022–2024 "15T-token era" is a direct consequence.
- **Basis for downstream prediction.** Llama 3 extends Chinchilla from compute → loss to compute → downstream accuracy (see [downstream-scaling-laws](downstream-scaling-laws.md)).
- **Explains the "7B era."** Models like Mistral 7B and Llama 3 8B get enormous capability from being trained past compute-optimal. For small models, over-training is often the most valuable lever.

---

## Gotchas & tricks

- **Chinchilla is an empirical law, not a theorem.** The exponents α ≈ 0.34, β ≈ 0.28 are fitted constants. They drift with architecture, data mix, and scale. Expect 5–10% deviations at very different regimes (small models, specialized domains).
- **The `6·N·D` FLOP formula ignores attention.** Attention FLOPs scale with `B · S² · H` — at long context this is a meaningful fraction of total compute. Chinchilla was trained at S = 2048; at S = 128k the 6·N·D approximation breaks down. Modern long-context runs need corrected FLOP counts.
- **The "20:1" rule is compute-optimal, not deployment-optimal.** For inference-deployed models, over-training (ratio of 100:1, 1000:1) is common and usually worth it. Chinchilla gives you the floor.
- **Doesn't handle mixture-of-experts.** N = "active parameters" or "total parameters" in an MoE? Chinchilla's derivation assumes dense. For MoE, use active-parameter counts as a first approximation; DeepSeek-V3 and Mixtral both inform this choice.
- **Data quality is not in the formula.** Chinchilla assumes roughly-uniform-quality data. If your data mix changes, the effective `β` changes. This is why Llama 3 and others do scaling-law experiments over **their own** data mix (see [downstream-scaling-laws](downstream-scaling-laws.md)).
- **Loss isn't the only thing.** Chinchilla optimizes validation loss. Downstream benchmark accuracy doesn't follow exactly — the relationship between loss and accuracy is task-dependent. Llama 3's extension to downstream-task scaling fills this gap.
- **Doesn't tell you when over-training stops helping.** Empirically, over-training continues to improve small models for *at least* 1000:1 ratios. No Chinchilla-style law predicts where that asymptote lies.
- **Doesn't account for LR schedule effects at the edges.** Chinchilla's own Approach 3 is immune to schedule-calibration bugs (it fits parameterically); but Kaplan-style Approach-1 studies are not.
- **Bigger batch size loosens the law slightly.** Larger batches have worse per-step efficiency; the compute-optimal N shifts slightly. Negligible for practical regimes.

---

## Sources

- Paper: *Training Compute-Optimal Large Language Models* — Hoffmann et al., DeepMind, 2022, arXiv 2203.15556 — the Chinchilla paper. Three approaches, parametric loss, empirical validation.
- Paper: *Scaling Laws for Neural Language Models* — Kaplan et al., OpenAI, 2020, arXiv 2001.08361 — the precursor whose LR-schedule artifact Chinchilla corrected.
- Paper: *Broken Neural Scaling Laws* — Caballero et al., 2022, arXiv 2210.14891 — refinements showing the power-law breaks at very small or very large scales.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — extends Chinchilla to compute-optimal *benchmark accuracy* prediction (see [downstream-scaling-laws](downstream-scaling-laws.md)).
- Blog: *Chinchilla's wild implications* — nostalgebraist, 2022 — widely-referenced unpacking of what Chinchilla means for the field.
