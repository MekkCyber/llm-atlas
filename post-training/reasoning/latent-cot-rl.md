# Switchable Latent CoT with On-Policy RL

*Depth — latent chain-of-thought that the model enters and exits via explicit boundary tokens, making the latent block trainable with on-policy RL and inspectable for mechanistic interpretability.*

**TL;DR:** Latent chain-of-thought (continuous hidden-state recurrence between input and output) has historically been hard to train with on-policy RL (the latent isn't a sampled discrete action) and hard to interpret (no anchor for probing). **Switch** introduces a single pair of discrete boundary tokens — `<swi>` to enter latent mode and `</swi>` to exit — that fix both. Because the boundaries are sampled like any other token, the **GRPO policy ratio is well-defined**, and the same boundaries serve as causal-intervention anchors for mechanistic analysis. Trained with a visible-to-latent curriculum and a Switch-GRPO objective that propagates gradients through the recurrent latent computation.

**Prereqs:** [../grpo](../grpo.md), [long-cot-rl](long-cot-rl.md)
**Related:** [../_rl](../_rl.md), [../../interpretability/README](../../interpretability/README.md)

---

## What it is

Latent CoT replaces visible reasoning traces with **continuous hidden-state recurrence**: the model loops its own hidden state through additional forward passes between input and output instead of emitting reasoning tokens. The hope is to compress reasoning and avoid the cost of long visible chains. The practical problems have been:

- **RL training**: on-policy methods (PPO, GRPO) need a sampled action to compute a policy ratio. A continuous hidden state isn't a sampled discrete action, so the importance-sampling correction breaks.
- **Interpretability**: without discrete anchors, there's nothing to causally intervene on or probe.

Switch resolves both with two **discrete boundary tokens**: `<swi>` signals "enter latent recurrence", `</swi>` signals "exit". The latent block in between is still continuous, but the boundaries are real tokens sampled from the policy. The GRPO ratio is defined at every decision point (visible token or boundary), and the boundary tokens are natural foothold for probing.

## How it works

- **Vocabulary**: the model has two new tokens, `<swi>` and `</swi>`, added to the standard vocabulary.
- **Generation loop**: at each step the model emits either a visible token, `<swi>` (entering latent mode), or — if currently in latent mode — runs $k$ recurrent hidden-state passes and then emits `</swi>` to return to visible mode.
- **Switch-GRPO objective**: the GRPO policy ratio is computed at every visible-token and boundary-token sample. Gradients flow back through the recurrent latent computation via standard backprop-through-time. The KL penalty toward the reference model is computed at boundary tokens too — preventing the model from collapsing into all-latent or all-visible modes.
- **Visible-to-latent curriculum**: training starts with mostly visible reasoning (high cost on `<swi>`), then gradually relaxes the penalty so the policy learns to use latent computation where it's most useful.

Mechanistic analysis exploits the boundary tokens directly: probe the hidden state at the moment `<swi>` is emitted, intervene by suppressing or forcing it, and measure the downstream effect on accuracy.

## Why it matters

- **First formulation that makes latent CoT both RL-trainable and mechanistically inspectable.** Previous latent-CoT approaches either trained with maximum-likelihood (and missed the gains RL gives long-CoT models) or trained with awkward custom RL losses that broke the standard tooling.
- **Beats prior hidden-state-recurrence baselines at the same scale.** The boundary-token design isn't just engineering convenience — the policy learns a *sharply localized*, learned switching policy rather than emitting boundaries as stylistic decoration.
- **Opens an interpretability foothold for "how RL changes the model from the inside."** The boundary tokens are causal-probe targets, and the Switch paper's analysis finds the latent step performs problem-specific computation concentrated at a single hidden-state transition on entry.

## Gotchas & tricks

- **Boundary token usage rate** is a key knob. If the model rarely emits `<swi>`, you're not using latent computation; if it emits constantly, the visible trace becomes uninterpretable. The curriculum controls this.
- **Latent depth $k$** (number of recurrent passes per latent block) trades compute for capacity. Empirically a small number suffices for most steps.
- **Probing inside the latent block** is harder than probing at the boundaries — the boundary tokens are where mechanistic analysis is cleanest.
- **Watch the KL penalty** at boundary tokens carefully; a too-strong KL keeps the model in visible-only mode (the reference's mode).

## Sources

- Paper: Switch — Guo et al. (2026) — [arXiv:2606.13106](https://arxiv.org/abs/2606.13106)
