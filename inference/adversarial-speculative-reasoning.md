# Adversarial Speculative Reasoning
*Depth — using speculative decoding as a semantic routing lever for long-horizon agents, not just a speed optimization.*

**TL;DR:** Train the draft model to **be rejected at critical decision points** so that the target model authors those decisions itself. Standard speculative-decoding's exactness guarantee (token distribution preserved) holds throughout; what changes is **which tokens come from the target**. On long-horizon agentic coding tasks, redirecting critical decisions to the target lifts end-to-end success without altering the target or its distribution.

**Prereqs:** [speculative-decoding.md](speculative-decoding.md)
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

Classical speculative decoding trains the draft to **match** the target as closely as possible — higher match rate, higher acceptance, bigger speedup. This is a purely systems framing: all tokens are equivalent, so maximize acceptance.

For long-horizon agents, not all tokens are equivalent. A token that names which file to edit, which tool to call, or which branch of logic to take has outsized downstream impact. If that token happens to come from the draft and is accepted, the agent can inherit a small draft-side bias that compounds over many turns. Adversarial speculative reasoning flips the training objective of the draft: on tokens that matter, the draft should produce output the target would *reject*, forcing the target to write that token.

The target's token distribution is unchanged. The agent's trajectory distribution shifts because the identity of the writer at high-impact tokens has shifted.

---

## How it works

Two components are modified from baseline speculative decoding:

1. **Critical-point labeling.** During draft training, label a subset of tokens as *critical decisions* — typically tokens where perturbation in a held-out dataset produces large downstream behavior changes. In the paper's setup, these are tokens at tool-call boundaries and decision branches in agent trajectories.

2. **Adversarial draft training.** At critical-point tokens, train the draft $q$ to produce a token $y$ such that the target's probability $p(y)$ is low — i.e. $y$ will be rejected by the standard speculative-decoding rejection step. At non-critical tokens, train the draft normally (match the target).

At inference, speculative decoding proceeds unchanged — draft $K$, target verifies $K+1$ in parallel, accept/reject per position with the standard rule. Critical-point tokens almost always fail the acceptance test; the residual distribution $p_{\text{resid}}(y) \propto \max(0, p(y) - q(y))$ is sampled, which is nearly $p$ itself at those positions (because $q$ is concentrated where $p$ is low). The target effectively writes that token.

No change to the target model, no change to the serving stack beyond draft weights. All exactness guarantees of speculative decoding carry over because this is the standard algorithm applied to a different draft.

---

## Why it matters

- **Quality gain from an inference-time trick.** No retraining of the target, no change to the agent prompt or loop. Just a differently-trained draft.
- **Reframes speculative decoding.** Draft training becomes a two-objective problem (match where cheap, mismatch where critical), and the critical-point definition is a design lever that encodes what the system cares about.
- **Composable with everything.** Works on top of EAGLE/Medusa-style self-speculation (train the extra heads adversarially at critical points) and under batching.
- **Related to Route Anything's insight at token scale.** Where Route Anything routes between models per request, this routes between models per *token* — with exactness as a side benefit.

---

## Gotchas & tricks

- **Critical-point labeling is load-bearing.** Overly aggressive labels push most tokens back to the target and erase the speedup; underly aggressive labels erase the quality gain. Calibrate on a held-out trajectory set.
- **Speedup drops where quality gain lives.** By construction, critical-point tokens come from the target — each is a full target forward pass (no acceptance). The net speedup is lower than vanilla speculative decoding.
- **The draft still matters at non-critical tokens.** Train it well there; a draft that is only adversarial is just a slow autoregressive run.
- **Not a free quality boost unconditionally.** On short-horizon tasks the critical-point labels have little signal and the method reduces to vanilla speculative decoding.
- **Target distribution is preserved by the algorithm, not by the labels.** The labels change which tokens come from where; they do not affect the exactness proof of the underlying speculative-decoding step.

---

## Sources

- Paper: *Adversarial Speculative Reasoning Improves Long Horizon Agents* — Luo, Fang et al. (Subconscious Systems, MIT, Princeton, Northeastern), 2026, arXiv 2611.05345 — primary source.
- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022, arXiv 2211.17192 — baseline algorithm whose exactness proof carries over.
