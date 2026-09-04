# Sparse Prefilling (CRISP)

*Depth — input-adaptive sparse attention at prefill time, with structural-mass routing and a sink-aware coverage threshold.*

**TL;DR:** Prefill is the quadratic phase of long-context inference. Dynamic sparse-attention methods route each attention head to a sparse pattern per prompt, but two bottlenecks recur: routing computed via JSD/KL over pooled attention costs a matmul and a divergence per head, and strict cumulative-coverage thresholds accumulate O(N) background noise at long context. **CRISP** (Nguyen et al., 2026) replaces JSD routing with **C_struct** — a structural proxy that reads routing decisions off Vertical-Slash-compatible positions — and replaces cumulative-coverage with a **sink-aware threshold** grounded in the noise floor. Result: strongest sparse method on InfiniteBench / RULER / LongBench across two model families, **matches or beats dense attention on retrieval tasks**, and **up to 5.30× attention speedup at 512k tokens**.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md), [_sparse-attention](_sparse-attention.md)
**Related:** [declarative-attention](declarative-attention.md)

---

## What it is

Sparse prefilling makes each attention head compute attention over only a subset of the input at prefill. CRISP is a dynamic scheme — the subset depends on the *input prompt* — that fixes two structural gotchas the previous generation of dynamic methods (MInference, SnapKV-class) baked in.

## How it works

### 1. Structural-mass routing (C_struct)

Prior dynamic routing (MInference-style) compares a head's pooled attention map to a set of reference patterns (Vertical, Slash, Block-Sparse) via Jensen–Shannon divergence. Two costs: the pooled matmul that produces the map, and the KL divergences per pattern per head.

C_struct observes that the routing decision only depends on where mass lies relative to Vertical / Slash positions — not on the fine-grained distribution. It computes only the *aggregate mass sitting at Vertical-Slash-compatible positions* directly from the softmax output, reproducing JSD's routing decision without the pooled matmul or the KL step.

### 2. Post-softmax mass cliff and sink-aware threshold

Coverage-based selection ("keep the top tokens whose softmax mass sums to ≥ p") has an implicit assumption: mass is concentrated on the tail. At long context, that fails. The tail of the softmax carries near-uniform noise whose *cumulative* mass grows O(N) — a strict threshold like p = 0.95 pulls in O(N) noisy tokens at 512k and the selection loses signal.

CRISP formalizes this as the **post-softmax mass cliff**: mass drops sharply from the top-k signal region to the noise floor. It then selects tokens by thresholding *above the noise floor* — a sink-aware criterion that automatically scales with context length instead of accumulating noise as N grows.

### 3. What actually runs at prefill

For each layer / head at prefill:

1. Compute the (approximate) attention map.
2. Read C_struct off the map → route to `{ Vertical, Slash, Block-Sparse, Dense }`.
3. Select tokens above the noise floor for the chosen pattern.
4. Run the sparse attention kernel on the selected KV.

Steps 1–3 are the "selection budget"; step 4 is the actual attention. CRISP's win is that steps 1–3 do less work AND produce cleaner selections, so step 4 both runs faster and retains accuracy.

## Why it matters

- **Two independent wins, one framing.** Both improvements come from the same observation — that post-softmax mass has structure the prior generation ignored — which makes the recipe simpler to implement and reason about than two orthogonal knobs.
- **Retrieval-heavy tasks stop being the sparse-attention Achilles heel.** Prior dynamic methods routinely lost 10–30 pp on retrieval tasks vs. dense; CRISP matches or exceeds dense there. That flips the deployment calculus for RAG-style workloads.
- **The speedup grows with context.** 5.30× at 512k is driven by the O(N) noise elimination — the longer the context, the more of the previous method's compute was pure background. This is the right shape for the direction long-context inference is heading.
- **Cheap to slot into existing kernels.** C_struct produces the same routing labels as JSD, and the noise-floor threshold plugs into any selection stage that currently takes a coverage parameter — CRISP is close to a drop-in improvement.

## Gotchas & tricks

- **The noise-floor estimator needs a few tokens of runway.** For very short prompts (well below the point where the cliff structure appears) the threshold degrades to top-k with an implicit k; fall back to dense under a threshold length.
- **Cannot help with decode-time cost by itself.** CRISP addresses prefill quadratic cost, not the per-step decode read. Pair with a decode-time sparse method (e.g. [declarative-attention](declarative-attention.md)) for end-to-end wins.
- **Vertical-Slash bias.** C_struct's routing signal is defined against Vertical / Slash / Block patterns — a workload with attention patterns outside that vocabulary can be misrouted. The paper's evaluation covers standard long-context benchmarks; unusual attention geometries deserve profiling.
- **Sink-aware ≠ StreamingLLM sinks.** CRISP's "sink" refers to the noise floor of the softmax distribution, not the first-token attention sink of StreamingLLM. The names overlap; the mechanisms don't.

## Sources

- Paper: *CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling with Structural-Mass-Motivated Routing* — Huu Huy Nguyen, Chien Van Nguyen, Franck Dernoncourt, Ryan A. Rossi, Linh Ngo Van, Jieyang Chen, Thien Huu Nguyen — 2026 — [arXiv:2609.01925](https://arxiv.org/abs/2609.01925) — Adobe Research · University of Oregon.
- Related: *MInference* — Jiang et al., 2024 — the Vertical-Slash-A-shape pattern vocabulary CRISP builds on.
