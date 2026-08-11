# Modular TTT
*Depth — DAG-based framework for composing test-time-training inner learners.*

**TL;DR:** Test-time training (TTT) treats sequence modeling as an *online* learning problem: fast weights are updated by an internal learning rule at inference time. Prior TTT variants (DeltaNet, Gated DeltaNet, etc.) each hard-code the inner learner, making it hard to isolate what actually matters. Modular TTT exposes the inner learner as a directed acyclic graph — fast-weight network, loss, LR, weight decay, normalization all as swappable primitives — then ablates them systematically. Best variant matches Gated DeltaNet at 410M / 1.45B on 100B tokens.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [README.md](README.md)
**Related:** [_test-time-training.md](_test-time-training.md), [transformer-block.md](transformer-block.md)

---

## What it is

TTT variants (DeltaNet, Gated DeltaNet, RWKV-7, TTT-Linear) all share the pattern of "inner learner updating fast weights per token," but each ships with a specific choice of network, loss, and update rule. Modular TTT is a framework that represents the inner learner as a DAG of primitive operations — *train-view forward*, *train-view backward*, *causal query-view* — with the design axes exposed as explicit knobs.

Once the framework can realize any of the existing variants, you can strip axes down to a controlled ablation.

## How it works

The inner-learner DAG factors into three primitive rule sets:
1. **Train-view forward:** how the fast-weight network processes the current token.
2. **Train-view backward:** how the loss gradient flows back to update fast weights (the inner learning rule).
3. **Causal query-view:** how the current query reads out from the updated fast weights.

Explicit design axes exposed at the primitive level: fast-weight network topology, inner-loss function (MSE, inner-product, etc.), inner-loop learning rate, weight decay, normalization, and fast-weight state transition rule.

The framework then composes these into the full graph-level TTT computation. Every existing variant is one setting of the DAG.

**Ablation findings.** With the framework in hand, the paper reports:
- Small LR init helps.
- Weight decay helps.
- A single-layer nonlinearity in the fast-weight net helps.
- MSE and inner-product losses perform ~equivalently.
- **Deeper** fast-weight networks and normalization **hurt** (excessively large activations).
- Residual connections and gating add little measurable benefit.

The resulting best-variant, trained at 410M and 1.45B params on 100B tokens, matches Gated DeltaNet on training loss and downstream benchmarks.

## Why it matters

- **Turns TTT from a swarm of variants into a design space.** Same value the LayerNorm-vs-RMSNorm-vs-... ablation grid brought to normalization.
- **Falsifies "deeper fast-weight = better."** A common architectural intuition that turns out to hurt because it induces activation blowup.
- **Sets a reproducible substrate.** Future TTT variants can be described as DAG configurations, not as new codebases.

## Gotchas & tricks

- **Activation blowup is the failure mode.** Deep fast-weight nets or aggressive normalization drive activations out of range — instability, then divergence.
- **Ablation findings scale-dependent.** Reported at 410M / 1.45B on 100B tokens; larger scales may re-rank.
- **Framework overhead ≠ zero.** Being able to swap primitives costs some kernel-fusion opportunity vs. hand-tuned baselines.
- **Doesn't cover softmax-attention transformers.** TTT and standard attention are architecturally distinct; Modular TTT is a substrate for the TTT family only.

## Sources

- Paper: *Modular TTT: Rethinking Test-Time Training as Composable Modules* — Tang, Qin, Pan, Li, Liu, Zhang — SJTU / Shanghai Innovation / ByteDance Seed, 2026 — arXiv:2608.07110.
- Prior: *Test-Time Training with Self-Supervised Learning* — Sun et al., 2020 — original TTT framing.
- Prior: *Gated DeltaNet* — Yang et al., 2024 — the strongest existing TTT variant, matched by the best Modular TTT configuration.
