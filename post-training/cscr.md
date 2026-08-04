# CSCR — Counterfactual Sensitivity Credit Reallocation
*Depth — smoothing GRPO's per-token credit by down-weighting the tokens that would swing the outcome the most.*

**TL;DR:** Critic-free RL (GRPO and its descendants) converts a response-level reward into a single scalar and broadcasts it equally to every token in the rollout. In long chain-of-thought reasoning that's a bad approximation — a handful of pivotal tokens carry most of the semantic weight, and giving them the same credit as glue tokens distorts the gradient. CSCR estimates each token's **counterfactual sensitivity** on the final outcome, reduces credit for the most-sensitive tokens, and renormalizes token-level advantages while preserving the verifier's assigned sign.

**Prereqs:** [grpo.md](./grpo.md), [long-cot-rl.md](./reasoning/long-cot-rl.md)
**Related:** [prm.md](./reasoning/prm.md), [cripo.md](./cripo.md)

---

## What it is

A drop-in modification to GRPO's advantage broadcast that acknowledges long-CoT credit assignment is uneven. The intuition:

- Uniform broadcast pretends every token contributes equally to the outcome.
- The true contribution is unimodal — a few tokens are decisive, the rest are supporting.
- Without correction, those few tokens dominate the gradient and everything else is noise around them.

CSCR is a lightweight, critic-free way to correct this without training a PRM.

## How it works

1. **Estimate counterfactual sensitivity per token.** For each token `t` in the rollout, approximate how much perturbing it would swing the final answer (e.g., by resampling from a small-radius replacement distribution and checking outcome flip probability). Denote this `s_t`.
2. **Down-weight high-sensitivity tokens.** Compute a per-token attenuation factor that is decreasing in `s_t` — the most-sensitive tokens get the smallest weight.
3. **Renormalize.** Rescale the down-weighted per-token advantages so their sum matches the original response-level advantage (preserves the verifier's sign and magnitude at the response level).
4. **Optimize** as usual with the corrected per-token advantages.

## Why it matters

- Fixes a specific, well-defined pathology of GRPO in long-CoT that other tricks (length penalty, entropy bonus) only address indirectly.
- Doesn't require a separate reward model or process supervision, so it composes with any RLVR setup.
- Consistent improvements over GRPO/DAPO-style baselines on math reasoning; larger gains where CoT is long.

## Gotchas & tricks

- Sensitivity estimation is the cost driver; simple approximations (single-token resample + majority vote among a few samples) work in practice.
- Down-weighting is intentional — it seems backwards, but it flattens the gradient dominated by pivotal tokens so *supporting* tokens actually receive signal too.
- Preserving the response-level sign after renormalization matters: if you drop it, CSCR can silently reverse the verifier's judgment.

## Sources

- Paper: *Not All Tokens Deserve Equal Credit: Counterfactual Sensitivity Credit Reallocation for Long-CoT Reasoning* — He, Wu, Wang, 2026 — [arXiv:2607.27888](https://arxiv.org/abs/2607.27888)
