# VIA-SD — Verification via Intra-Model Routing

*Depth — a multi-tier speculative-decoding scheme that adds a slim, routed verifier between "accept" and "full recompute".*

**TL;DR:** Standard [speculative decoding](speculative-decoding.md) is binary: a draft token is either accepted by the full verifier or it triggers a full recomputation. VIA-SD inserts a *middle tier* — a slim sub-model derived from the verifier via intra-model routing — that handles medium-confidence tokens cheaply. Across four tasks and several model families, rejection rate drops by **0.10–0.22** and end-to-end speed improves **10–20%** over strong SD baselines, **2.5–3×** over non-drafting decoding, without modifying SD training.

**Prereqs:** [speculative-decoding.md](speculative-decoding.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md)

---

## What it is

A drop-in upgrade to the speculative-decoding verification step. The accept / reject decision is replaced with a three-way routing:

| Confidence band | Action | Cost |
| --- | --- | --- |
| High | Accept draft outright | ≈ 0 |
| Medium | Re-generate via slim verifier (intra-model routed path) | ~ fraction of full |
| Low | Full verifier recomputation | full |

The slim verifier is **not a separate trained model**. It's a routed sub-path of the full verifier — e.g. fewer active experts in an MoE, or a subset of attention heads — derived at inference time. This means no extra training pipeline and no extra checkpoint to store.

## How it works

```
draft = drafter.generate(prefix, k tokens)
for token x_i in draft:
    c = confidence(x_i)            # cheap signal from drafter or first verifier layers
    if c >= τ_high:
        accept(x_i)
    elif c >= τ_low:
        x_i' = slim_verifier(x_i)  # routed sub-path of full verifier
        if matches(x_i, x_i'):
            accept(x_i)
        else:
            accept(x_i')           # cheaper than full recompute
    else:
        full_verifier_recompute(x_i)
```

The thresholds $τ_{high}, τ_{low}$ are tuned per model family. The slim verifier is implemented as a subset routing through the full verifier's parameters — for an MoE, a smaller top-$K$; for a dense model, an attention-head or layer-skip routing.

## Why it matters

- Pulls the verifier's expected cost towards the *expected difficulty* of the token instead of paying full cost on every rejection.
- Removes a long-standing brittleness in standard SD: rejection at position $j$ wastes the verifier's compute for tokens $j+1..k$. With a recoverable middle tier, more of that work survives.
- Compatible with existing SD frameworks (no training-side changes), so it composes with EAGLE / Medusa / tree drafting.

## Gotchas & tricks

- **Calibration of $τ_{high}, τ_{low}$ matters more than the slim model itself.** Too aggressive on $τ_{high}$ and you accept bad tokens; too conservative and you collapse back to binary SD.
- **Slim verifier path is implementation-dependent.** On MoE backbones the natural choice is fewer top-$K$ experts; on dense backbones, head-pruning or early-exit. The paper covers multiple instantiations.
- **Speedup compounds with tree drafting.** The 10–20% headline is over linear-draft SD; expect more on top of tree-draft baselines (EAGLE-2, Medusa-v2).
- **No quality loss** when calibrated correctly, because uncertain tokens still fall through to the full verifier. The slim middle tier is a *recovery* path, not a *fallback* path.

## Sources

- Paper: *VIA-SD: Verification via Intra-Model Routing for Speculative Decoding* — Xian, He, Xu, Yang, 2026, [arXiv:2606.12243](https://arxiv.org/abs/2606.12243).
- Background: [speculative-decoding.md](speculative-decoding.md).
