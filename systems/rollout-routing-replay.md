# Rollout Routing Replay (R3)
*Depth — record MoE expert choices during rollout, replay them in the trainer's forward pass, so rollout↔trainer stay consistent under non-deterministic routing.*

**TL;DR:** vLLM (rollout engine) and Megatron (trainer) can select **different experts** for the same tokens even with identical weights — kernel differences, numerical noise, and precision differences all contribute. In GRPO-style on-policy RL this breaks the policy-gradient equivalence between rollout and training. **R3** records the selected experts per `(token, MoE layer)` during rollout and replays those selections in the trainer. Storage cost: ~**20 MB per sample** as `uint8` tensors. In Axion 1.0, mean gradient norm drops from 1.998 → 1.922 and mean PPO KL drops from 0.0413 → 0.0396. Benchmark effects are mixed, and the paper is careful to note that staleness and context-condensation also affect the log-prob diagnostics.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [partial-rollouts.md](partial-rollouts.md) · [../case-studies/axion-1-0.md](../case-studies/axion-1-0.md)

---

## What it is

In an MoE model, the router at each layer picks top-$K$ experts per token via a non-smooth argmax over affinity scores. Two equivalent-weights implementations (rollout engine vs trainer) can disagree on which experts tie-break at the top-$K$ boundary because:

- Kernel implementations compute logits in slightly different orders (fused vs unfused matmul, different GEMM libraries).
- Precision differs between inference (often FP16/BF16 with specific kernels) and training (BF16 with different reduction order).
- Numerical noise in the router logits can flip a tie.

When this happens in on-policy RL, the "policy" that generated the rollout differs from the policy the trainer updates — small per-token differences that aggregate into biased gradients and unstable KL estimates.

---

## How it works

Two additions to the standard GRPO loop:

1. **Record during rollout.** The rollout engine logs the chosen expert indices for every `(token, MoE layer)` as a `uint8` tensor per sample (one byte per expert choice is sufficient for expert counts $< 256$). The paper reports ~20 MB per sample of routing records.

2. **Replay in trainer.** The trainer's forward pass **overrides the router's top-$K$ selection** with the recorded indices. The expert gate weights are still computed from the trainer's live router (so the gradient flows through the router), but the identity of the active experts is forced to match the rollout.

The rest of the update — GRPO advantage, PPO-clipped loss, KL-to-reference — is unchanged.

### Why it's correct

Replaying the expert selection preserves on-policy semantics: the gradient the trainer computes is the gradient of the *same* MoE forward pass the rollout saw. The gate-weight computation still flows gradients to the router, so the router can still learn — it's only the top-$K$ choice at this particular step that is forced.

---

## Why it matters

- **Removes a specific non-determinism source from MoE+RL training.** Per-token MoE routing drift is subtle; it won't show up as a loss spike, but it pollutes advantages and KL measurements.
- **Cheap.** 20 MB per sample is small compared to the hidden state footprint; `uint8` tensors compress well on disk.
- **Composable.** Works with aux-loss-free balancing, with any router (sigmoid or softmax), and with any top-$K$. No change to the model architecture.
- **Diagnostic-grade gains.** The Axion paper reports ~4% drops in mean gradient norm and PPO KL at 122B — modest, but these diagnostics are the kind that correlate with training stability over long RL runs.

---

## Gotchas & tricks

- **Benchmark effects are mixed.** In Axion's ablations, HLE-130 and FrontierScience Research improved; SEAL-0 dropped from 63.1% → 58.6%. The paper explicitly cautions that log-probability diagnostics "also reflect asynchronous staleness and context-condensation effects, which limit causal attribution." R3 is a stability patch, not a quality lever on its own.
- **Only helps when rollout and trainer actually disagree.** If both run the same kernel stack (uncommon in production), the routing already matches and R3 is a no-op cost.
- **Replayed selections freeze the router at those positions.** The router's gradient at a replayed step only updates its *gate weights*, not its *rank order*. Over many steps this is fine; in a single step it's a mild restriction.
- **Storage cost adds up at trajectory scale.** 20 MB × many thousands of samples per RL epoch is large. Stream to disk or use ring-buffer caching.
- **Doesn't remove all rollout/trainer mismatch.** Attention kernel differences, weight-sync lag, numerical drift in BF16 reductions — these remain. R3 is specifically for MoE routing.
- **Not relevant for dense models.** Only matters where routing introduces a non-smooth discrete choice.

---

## Sources

- Paper: *Axion 1.0: Open Agent Foundation Models, Simplified* — 2026, arXiv 2611.03233 — primary source (R3 introduced in §5.2 with ablations).
- Related: [../architectures/_moe.md](../architectures/_moe.md) — MoE routing mechanics.
- Related: [../post-training/grpo.md](../post-training/grpo.md) — the RL update R3 keeps consistent.
