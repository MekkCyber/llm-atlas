# DSpark
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A speculative-decoding framework that unifies two production-scale fixes: a **semi-autoregressive drafter** (parallel backbone + lightweight sequential module) that injects intra-block token dependencies to fight suffix decay, and **confidence-scheduled verification** that sets per-request verification length from estimated prefix-survival probabilities and the serving system's throughput profile. Deployed in DeepSeek-V4 serving; delivers 60–85% faster per-user generation vs the MTP-1 baseline at matched throughput.

**Prereqs:** [../pre-training/mtp.md](../pre-training/mtp.md)
**Related:** [_speculative-decoding.md](./_speculative-decoding.md), [confidence-scheduled-verification.md](./confidence-scheduled-verification.md), [self-speculation-decoding.md](./self-speculation-decoding.md)

---

## What it is

Parallel speculative-decoding drafters (Medusa, EAGLE-2, MTP-multihead) generate long candidate sequences in a single pass, but their accepted length decays fast: without intra-block dependencies, later tokens in the draft are guesses conditional on earlier *guesses*, so acceptance drops geometrically. Under high-concurrency serving, verifying long low-quality drafts wastes batch capacity on tokens with high rejection risk — throughput collapses.

DSpark fixes both the drafter and the verifier scheduling.

## How it works

### 1. Semi-autoregressive drafter

The drafter is a hybrid:

- **Parallel backbone** — proposes $K$ candidate tokens in one forward pass (like MTP-K or Medusa).
- **Lightweight sequential module** — a small AR head that runs *conditioned on each parallel proposal* to inject intra-block dependencies. Cheap because it operates on the compressed hidden state, not the full backbone.

This gives you the throughput of parallel drafting with the acceptance-length characteristics closer to sequential drafting — the "semi" is the compromise.

### 2. Confidence-scheduled verification

The verifier (main model) decides *how many* of the draft tokens to verify per request based on:

1. **Prefix survival probability.** For each draft position $i$, estimate $P_i$ = probability the first $i$ draft tokens all get accepted, from draft-side confidences.
2. **Throughput profile of the engine.** How much a longer verification hurts batch capacity at the current load.

Combine these into an optimal verification length $k^*$ per request: verify only up to the point where the marginal accepted token is still net-positive under current load. Under low load, $k^*$ is large (worth verifying long drafts); under high load, $k^*$ shrinks to conserve batch capacity for other requests.

## Why it matters

- **60–85% per-user speedup vs MTP-1 baseline** in DeepSeek-V4 production. MTP-1 was the previously-strong production baseline (single-token MTP draft); DSpark's semi-AR drafter + adaptive verification opens a new Pareto point.
- **Load-aware inference.** Standard speculative decoding treats draft/verify length as a fixed hyperparameter. DSpark makes it a per-request online decision — closer to how large-scale serving actually operates.
- **New latency tiers.** Enables interactivity SLOs (per-user tokens/s at target concurrency) that were previously unreachable.
- **Open-source.** Checkpoints + DeepSpec training repo released.

## Gotchas & tricks

- **The sequential module needs its own KV state** during drafting; not free in memory. Kept small to keep drafting cheap.
- **Verification length is dynamic** — the serving system must support variable-length verify batches, not a fixed $K$. Requires kernel-level support (custom masked-attention or padding + attention masks).
- **Prefix-survival estimates rely on drafter confidence being well-calibrated.** Poor calibration → over-optimistic long verifies → wasted compute. Calibration is part of training, not just an inference-time trick.
- **Not a drop-in for arbitrary base models.** DSpark is trained jointly with the target (verifier) model; you can't slap it on a checkpoint without at least a lightweight distillation stage.
- **MTP-K remains competitive** for pure per-user latency at low concurrency — DSpark's advantage is under load.

## Sources

- Paper: *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* — Yu, Shao, Li, et al., DeepSeek-AI / Peking University, 2026 — [arXiv:2607.05147](https://arxiv.org/abs/2607.05147).
- Code: DeepSpec — algorithm-driven training repo for speculative decoding.
