# DNeo
*Depth — parallel draft models for speculative decoding, done right.*

**TL;DR:** Parallel drafters (predict `k` future tokens in one shot from the target's hidden state) have historically lost to sequential drafters on acceptance. DNeo flips that by rethinking how the drafter encodes the target's features — **Mixture-of-Features Attention** across layers + a **rejection memory** of prior misses — giving ~1.3× end-to-end speedup over DFlash-2 at the same drafter size.

**Prereqs:** [../pre-training/mtp](../pre-training/mtp.md), [../architectures/multi-head-attention](../architectures/multi-head-attention.md)
**Related:** (inference/ is still thin — this page helps seed it)

---

## What it is

A **parallel-drafter architecture** for speculative decoding. A parallel drafter takes one forward pass of the target model's hidden state at position `t` and emits `k` draft tokens at positions `t+1, t+2, …, t+k` in parallel, which the target then verifies in one forward pass. The structural challenge is that the drafter has no causal signal between its own `k` outputs, so quality collapses relative to a sequential drafter that regenerates after each verified token.

## How it works

Three pieces, all inside the drafter:

**1. Mixture-of-Features Attention (MFA).**
The usual parallel drafter reads only the target's last layer at position `t`. MFA lets the drafter attend across **multiple target layers** of the same position with learned per-head weights. Early-layer features often carry local-syntax cues the last layer has discarded — critical for predicting `t+2, t+3`.

```
target_features[t]  = concat(h_L[t], h_{L-k}[t], …, h_1[t])   # all layers
drafter_q           attends → MFA(target_features[t])
→ k parallel draft tokens
```

**2. Rejection memory.**
After each verify step, the target reports which of the drafter's `k` tokens were rejected. DNeo stores these rejection positions (and the token the target actually produced) in a small FIFO memory and conditions the drafter's next call on that memory. The drafter learns to avoid its own characteristic failure positions.

**3. Feature-position encoding fixes.**
Standard parallel drafters reuse the target's positional encoding for all `k` slots. DNeo uses a dedicated per-slot encoding that distinguishes "I am drafting position `t+j`" from "I am drafting position `t+1`" — the drafter stops conflating them.

## Why it matters

Parallel is the *right* asymptotic for speculative decoding: one target forward pass covers both the hidden state supply and the verification, and `k` draft slots don't cost `k` draft forwards. But every parallel drafter until now has underperformed a sequential drafter of the same size. DNeo closes that gap — **~1.3× faster than DFlash-2** at matched drafter size and no quality regression — which makes parallel the obvious default when a drafter has to coexist with large-batch serving (where sequential drafters' per-step overhead compounds).

## Gotchas & tricks

- **MFA needs the target's intermediate activations.** That's a change to the inference runtime, not just the drafter. vLLM/SGLang need a hidden-states-export hook for all target layers the drafter reads.
- **Rejection memory grows unbounded if left uncapped.** The paper uses a fixed FIFO of a few hundred positions; longer buffers help early-in-generation but add drafter latency.
- **Drafter training is on-policy against the target.** You can't drop in a DNeo drafter trained on one target into another target of similar size; the rejection-memory conditioning is target-specific.
- **Hidden-state access pattern matters.** On H100/B200 the extra memory reads for MFA are cheap; on bandwidth-starved hardware they are not.

## Sources

- Paper: *DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding* — Bergsma, Elhoushi, Zhang, Gray, Hestness, Dey, Fioretto (Cerebras / Syracuse), 2026 — [arXiv:2611.21925](https://arxiv.org/abs/2611.21925)
