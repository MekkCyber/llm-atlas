# Data Mixture
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Given a pool of pretraining data broken into $D$ domains (web, code, math, books, multilingual, …), *how much of each* should the model see? Data mixture is the most-empirical, highest-leverage lever in pretraining — a few percentage points of shift can beat entire architecture changes. Modern methods use **proxy models** (DoReMi, RegMix): sweep mixtures at small scale, extrapolate the winner to the big run. **CausalMix (2026)** reframes it as a **causal-inference** problem — the mixture is the treatment, downstream loss is the outcome — letting you *extrapolate* between different pool sizes without re-sweeping.

**Prereqs:** [_data-curation](_data-curation.md), [quality-filtering](quality-filtering.md)
**Related:** [deduplication](deduplication.md), [../pre-training/README.md](../pre-training/README.md)

---

## What it is

A pretraining corpus $\mathcal{P} = \bigcup_i \mathcal{P}_i$ decomposes into domains — web (CC), code (GitHub), math, books, arXiv, multilingual, wiki, ... A **mixture** is the sampling weights $w = (w_1, \ldots, w_D)$ with $\sum w_i = 1$ that decide the marginal token frequency each domain contributes to the training stream. The mixture appears in the loss as an expectation over domain-specific losses:

$$
L(w) = \sum_i w_i \cdot \mathbb{E}_{x \sim \mathcal{P}_i}\!\left[\ell(x; \theta)\right]
$$

Small differences in $w$ move downstream scores substantially — this has been documented since GPT-3's manual reweighting.

## How it works

Three generations of methods.

### 1. Manual heuristics (GPT-3, LLaMA)

Set weights by intuition (upweight books and code, downweight low-quality web) and validate at scale. Cheap but not principled — no reason the choice is optimal.

### 2. Proxy-model sweeps (DoReMi, RegMix)

Train many small proxy models (100M–1B) each on a different mixture, measure downstream loss, fit a **surrogate** $f(w) \to L$, and pick the argmin. Then train the target-scale model on that mixture.

- **DoReMi** (2023): use group DRO to learn worst-case-optimal mixture weights from a small proxy.
- **RegMix** (2024): regression over a random sample of mixtures; scales to many domains.

Works well when the target run's data distribution matches the proxy sweep's pool.

### 3. Causal-inference framing (CausalMix, 2026)

The proxy-sweep assumption breaks when the data pool changes (adding a new domain, filtering old data): re-fit the surrogate from scratch.

**CausalMix** frames the setup as CATE (conditional average treatment effect) estimation:

- **Covariates** — statistical features of the pool (per-domain size, quality metric, dedup ratio, …).
- **Treatment** — the mixture $w$.
- **Outcome** — post-training loss / downstream score.

Fit a causal model over many proxy runs (e.g., 512 Qwen2.5-0.5B runs across mixture-and-pool configurations), then *extrapolate* the optimal $w$ to a new pool without new proxy training. Reported: extrapolation from 0.5B proxy runs to a 7B model on an 800K-datapoint pool, and generalisation to long-CoT data on Qwen3-4B-Base.

## Why it matters

- **Mixture is where scaling laws hide.** Chinchilla, Llama, DeepSeek scaling reports all note that a few-percent shift in mixture beats moderate architecture changes.
- **Compute leverage.** A well-fit surrogate turns hundreds of proxy runs into one big-model run, instead of guessing.
- **Pool-drift robustness (CausalMix).** As curation pipelines evolve (filtering thresholds change, new sources arrive) the data pool changes, and pre-CATE methods require re-sweeping. Causal extrapolation keeps proxy compute reusable.

## Gotchas & tricks

- Proxy → target transfer is not free — the *rank* of mixtures often transfers but the absolute optimum can shift with model scale.
- Mixture and quality-filtering are entangled: aggressive filtering makes domain sizes smaller and skews the optimum toward under-represented but high-quality domains.
- Downstream *task* mixture differs from *loss* mixture. Fitting to validation loss can under-weight domains that dominate specific downstream tasks (code, math).
- Long-CoT and reasoning-corpus mixtures behave differently — CausalMix's generalisation to Qwen3-4B long-CoT is a data point, not a proof.
- Multi-epoch effects: if some domains repeat, the effective mixture drifts from the sampling weights.

## Sources

- Paper: *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* — Xie et al., 2023 — group-DRO proxy method.
- Paper: *RegMix: Data Mixture as Regression for Language Model Pre-training* — Liu et al., 2024 — regression surrogate over random mixtures.
- Paper: *CausalMix: Data Mixture as Causal Inference for Language Model Training* — Tang et al., 2026 — [arXiv:2607.01104](https://arxiv.org/abs/2607.01104).
- Paper: *Data Mixing Made Efficient: A Bivariate Scaling Law for Language Model Pretraining* — Kang et al., 2024 — mixture x scale interaction.
