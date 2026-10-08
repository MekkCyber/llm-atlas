# Weight Refit for RL

*Depth — the operation of pushing updated policy weights from the trainer to the inference workers between RL steps; the dominant systems bottleneck at trillion-scale agentic RL.*

**TL;DR:** In agentic RL, rollouts and training are typically separated: a trainer owns the policy and gradient steps, a fleet of inference workers runs the rollouts. Every RL step the inference workers need the fresh policy — the *refit*. Full-dense refit at trillion parameters is a terabyte-per-step broadcast and dominates wall-clock. Modern systems (NeMo-DCR 2026) ship only the ~1% of weights that changed per step via a relay-tree topology, with bit-exact convergence to a full-dense refit.

**Prereqs:** [ray.md](./ray.md), [partial-rollouts.md](./partial-rollouts.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [dualpipe.md](./dualpipe.md), [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

An LLM RL step has three systems-level phases:

```
┌────────────┐      ┌──────────────┐      ┌──────────────┐
│  Rollout   │ ───▶ │   Training   │ ───▶ │   Refit      │ ───▶ (next step)
│ (inference │      │  (trainer    │      │ (push weights│
│  fleet)    │      │   cluster)   │      │  to fleet)   │
└────────────┘      └──────────────┘      └──────────────┘
```

For dense models up to ~70B, refit is a nuisance (seconds). For frontier MoE at 500B–1T, refit becomes the bottleneck: 1T parameters × 2 bytes/BF16 = 2 TB to broadcast every step. At 400 Gb/s per NIC, a naive all-reduce broadcast is tens of minutes.

Refit is distinct from checkpointing: checkpoints are durable disk writes done rarely; refit is in-memory shipment done per step.

## How it works

### What changes per step

Empirically, only ~1–5% of BF16 weight bits change per RL step at frontier scale (NVIDIA observes 1–3% for 30B–1T models). This is unsurprising: policy-gradient updates are small, the learning rate is low, and most of the gradient mass lives in a thin shell of parameters.

That means the *dense* refit is doing ~30× more work than the *information* refit.

### Delta-compressed refit (NeMo-DCR)

The pipeline:

1. Trainer step produces new weights `W_new`. Compute `Δ = W_new − W_old`.
2. Sparsify: send only positions where `|Δ| > ε`. Compress the index + magnitude.
3. Push via a **relay-tree topology** (trainer → k relays → N workers). Avoids the all-reduce fan-in on the trainer.
4. Each worker applies `W_new ← W_old + Δ`.

Bit-exactness is preserved by sending the exact-bit delta (no additional quantization) and applying it in BF16. The receivers end with the *same bits* as a full-dense refit — important, because any drift between workers accumulates across steps.

### Reported numbers

NeMo-DCR: 1T-parameter model at 3% change rate takes **150 seconds** to refit via the relay tree, vs **87.5 minutes** for a dense broadcast — ~35× speedup. 30B–1T models see 12–40× speedup at 3–5% change rates.

### Prior art

- **Parameter servers** (early distributed training) do a similar partial-push but without bit-exactness or the sparsity assumption.
- **Weight streaming** (vLLM, SGLang) sends dense weights per step; fine for < 70B dense, prohibitive at trillion scale.
- **Delta encoding of checkpoints** (Megatron) is a disk-time analogue, not an in-memory refit.

## Why it matters

- **Enables many RL steps per hour at trillion scale.** If each refit is 87 minutes, you can afford maybe 10 steps per day; at 150 seconds you can afford hundreds per day. Agentic RL depends on the fast-step regime.
- **Composes with FP4 rollouts.** The TRACE/TRIAGE line is making the rollout cheaper; delta-refit is making the sync cheaper. Together they move the systems bottleneck off the critical path.
- **Bit-exactness avoids drift.** With fp-approximate or lossy delta compression, workers would diverge from the trainer over tens of thousands of steps; NeMo-DCR deliberately keeps the delta exact.

## Gotchas & tricks

- **The 1% change rate assumption** is empirical; at higher learning rates or during warmup it can be 5–20%, in which case the compression ratio drops. Monitor and fall back to dense when needed.
- **Relay topology must be fault-tolerant.** A dead relay node blocks a subtree; retries and backup relays matter at large fan-out.
- **Alignment with optimizer state.** If the optimizer state (momentum, Adam moments) also needs to refit (e.g., for distilled teacher updates), delta-compression doesn't automatically apply — optimizer state has different change-rate statistics.
- **The sparsity pattern can be exploited.** If the top-moving parameters are predictable (e.g., LoRA ranks), you can pre-group them. NeMo-DCR doesn't do this; it treats the delta as unstructured.
- **Hardware bias.** The method shines on cluster NICs where latency per message matters; on single-node setups the baseline dense broadcast is cheap enough that delta compression doesn't pay for itself.

## Sources

- Paper: *NeMo-DCR: Bit-Exact Delta-Compressed Refit for Scalable Agentic RL at Trillion-Parameter Scale* — Li et al., NVIDIA + Aalto, 2026 — [arXiv:2610.08430](https://arxiv.org/abs/2610.08430).
- Related: *Megatron-LM* and *DeepSpeed* checkpoint schemes — delta on disk, not in-memory refit.
- Related: *vLLM / SGLang* — the inference-fleet side of the refit protocol.
