# Looped Models — Fixed-Point Recurrent LMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **looped** language model recurses the same block over and over per token position, trading parameter count for depth-via-time. The 2026 *Done Right, Part II* paper shows that treating the recurrence as **approaching a fixed point** unlocks big wins: truncated BPTT during training, terminal-KV sharing during decoding, cheap distilled prefill, and faster RL updates. Two training fixes — a **learned depth prior** and **orthogonal input injection** — are what make the fixed-point behavior work in practice.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [transformer-block.md](transformer-block.md)
**Related:** [mla.md](mla.md) · [../pre-training/_training-stability.md](../pre-training/_training-stability.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A looped LM applies a single block $f$ for $D$ recurrences at each token position. Earlier looped models (Universal Transformer, Huginn) treat $D$ either as fixed or as a broad prior to sample from at training. All four of training, decoding, prefill, and RL pay a cost in $D$: training backprops through all $D$ recurrences, decoding keeps $D$ KV states, prefill does $D$ forward passes, and RL replays all $D$.

*Looped Models Done Right, Part II* observes that if the recurrent state converges to a **fixed point** of $f$, then the trajectory to it stops mattering — only the fixed point does. That observation makes all four costs collapsible:

- **Training:** truncated BPTT past convergence loses negligible signal.
- **Decoding:** share the KV of the terminal recurrence across depth (terminal-KV sharing).
- **Prefill:** distill a student that skips to the fixed point (1.79× faster).
- **RL:** compute gradients from saved rollout *states* instead of re-replaying the whole trajectory (2× faster updates).

## How it works

- **Depth prior from feedback.** Instead of a hand-set broad depth distribution, learn one from prediction feedback with an entropy regularizer. The entropy term keeps the prior broad enough to enable KV sharing, while prediction feedback keeps supervision concentrated at the target depth. Fixed-depth training breaks KV sharing (nothing to share across); Huginn's hand-set broad prior dilutes target-depth supervision more than sharing needs. The learned prior sits in between.
- **Orthogonal input injection.** Existing injection schemes add the input to the recurrent state; this lets the state's component **along the input direction** amplify or cancel the injection unpredictably. Orthogonal injection subtracts that component first, so the injection is a clean addition in a direction orthogonal to the current state.
- **Shared terminal KV.** After the recurrence converges, store one KV for the terminal state and reuse it across depths of attention. Loss is near-zero in accuracy.

## Why it matters

- From 100M to 1.6B, learned prior + orthogonal injection lower perplexity at **every scale** relative to Huginn's prior and existing injection schemes.
- At **1.6B with a 3× smaller KV cache**, the learned-prior model matches fixed-depth downstream averages with the full cache.
- Makes recurrent / looped LMs competitive on **all four cost axes** (training, decoding, prefill, RL) — the long-standing argument against them.

## Gotchas & tricks

- **Fixed-point convergence is empirical.** Not all tokens converge in the same number of recurrences. Monitor convergence; outlier tokens can require more depth.
- **Orthogonal injection changes init.** The orthogonal-subtraction step modifies gradient flow early in training — expect slightly different warm-up behavior than naive injection.
- **Terminal-KV sharing is lossless only after convergence.** If the terminal recurrence hasn't converged, sharing introduces real error. Pair with a convergence check per token.
- **Composes with reasoning RL.** The 2× RL speedup comes from computing gradients at saved states — this is agnostic to the RL algorithm (GRPO, PPO, etc.) and complements algorithm-level savings.

## Sources

- Paper: *Towards Looped Models Done Right, Part II: Rethinking at Fixed Points* — Huang et al., Institute of Foundation Models, 2026 — [arXiv:2610.06833](https://arxiv.org/abs/2610.06833).
- Prior: *Huginn* (looped LM with a broad depth prior).
- Context: Universal Transformer — Dehghani et al., 2018 — early recurrent-transformer design.
